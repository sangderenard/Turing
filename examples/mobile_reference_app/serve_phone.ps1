param([int]$Port = 8000)
$Repo = (Resolve-Path "$PSScriptRoot\..\..").Path
$Root = Join-Path $Repo "build\mobile-reference\web"
if (-not (Test-Path -LiteralPath (Join-Path $Root "index.html"))) {
    throw "Web product is missing. Run: py -3.11 examples/mobile_reference_app/build_mobile.py --web"
}
$Address = (Get-NetIPAddress -AddressFamily IPv4 |
    Where-Object { $_.IPAddress -notlike "127.*" -and $_.PrefixOrigin -ne "WellKnown" } |
    Select-Object -First 1 -ExpandProperty IPAddress)
Write-Host "On the same Wi-Fi, open http://${Address}:${Port}/ on the iPhone."
Write-Host "Safari can add it to the Home Screen after loading."
py -3.11 -m http.server $Port --bind 0.0.0.0 --directory $Root
