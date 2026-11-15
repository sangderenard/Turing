# The agent accounts' saved passwords and how a program is started as one of them. Shared by Set-AgentBox.ps1 (which
# makes the accounts and saves the passwords) and Start-AgentDay.ps1 (which runs the agents).

# Each password is saved by Set-AgentBox with Export-Clixml: DPAPI, the operator's key, so it opens only for the operator.
function Get-AgentCredentialFile([string]$Name) {
    Join-Path (Join-Path ([Environment]::GetFolderPath('LocalApplicationData')) 'SpeakToMe\agent-accounts') "$Name.xml"
}

function Read-AgentCredential([string]$Name) {
    $file = Get-AgentCredentialFile $Name
    if (Test-Path -LiteralPath $file) { Import-Clixml -LiteralPath $file } else { $null }
}

# Start a program as an agent account with that account's own environment. Start-Process -Credential copies the caller's
# environment (HOME, TEMP, APPDATA and PATH pointed at the operator's profile: the first -OpenConsole on 2026-11-14 tried
# to write its history and $HOME\gemini.key into C:\Users\alber and was refused), and -UseNewEnvironment gives only the
# machine scope (USERNAME reads SYSTEM). A ProcessStartInfo whose Environment is never touched passes no environment
# block, so CreateProcessWithLogonW builds one from the account's own profile; it also opens a new console by default.
# Its command line is limited to 1024 characters, so longer input goes in on standard input (-Redirect).
function Start-AsAgent([pscredential]$Credential, [string]$File, [string[]]$Arguments, [string]$Directory, [switch]$Hidden, [switch]$Redirect) {
    $info = [System.Diagnostics.ProcessStartInfo]::new($File)
    foreach ($a in $Arguments) { $info.ArgumentList.Add($a) }
    $info.UserName = $Credential.UserName
    $info.Domain = $env:COMPUTERNAME
    $info.Password = $Credential.Password
    $info.LoadUserProfile = $true
    $info.UseShellExecute = $false
    $info.WorkingDirectory = $Directory
    $info.CreateNoWindow = [bool]$Hidden
    if ($Redirect) { $info.RedirectStandardInput = $info.RedirectStandardOutput = $info.RedirectStandardError = $true }
    [System.Diagnostics.Process]::Start($info)
}

# Run a PowerShell script as an agent account, hidden, the script given on standard input. Returns its exit code, its
# output (stdout then stderr) and whether it ran past the timeout (then it is stopped).
function Invoke-AsAgent([pscredential]$Credential, [string]$Script, [string]$Directory, [int]$TimeoutSeconds = 1200) {
    $process = Start-AsAgent $Credential (Join-Path $PSHOME 'pwsh.exe') @('-NoProfile', '-NonInteractive', '-Command', '-') $Directory -Hidden -Redirect
    $out = $process.StandardOutput.ReadToEndAsync()
    $err = $process.StandardError.ReadToEndAsync()
    $process.StandardInput.Write(($Script -replace "`r", '') + "`n")
    $process.StandardInput.Close()
    $timedOut = -not $process.WaitForExit($TimeoutSeconds * 1000)
    if ($timedOut) { try { $process.Kill($true) } catch { } ; $process.WaitForExit() }
    $text = ($out.Result.TrimEnd(), $err.Result.TrimEnd() | Where-Object { $_ }) -join "`n"
    [pscustomobject]@{ ExitCode = $(if ($timedOut) { $null } else { $process.ExitCode }); Output = $text; TimedOut = $timedOut }
}

Export-ModuleMember -Function Get-AgentCredentialFile, Read-AgentCredential, Start-AsAgent, Invoke-AsAgent
