<#
.SYNOPSIS  Prints the rollback from the state root to the legacy build\meowdoku layout. Moves and deletes nothing: Migrate-State.ps1 only copied, so the legacy tree is intact.
#>
Write-Host 'Rollback to the legacy layout (nothing is moved or deleted by this script):'
Write-Host '  1. Stop-Apache.ps1 (if running).'
Write-Host '  2. instances.json: remove the top-level "state_dir" key.'
Write-Host '  3. Remove the environment variable:  Remove-Item Env:\MEOWDOKU_STATE_DIR  (and from the persisted user/service environment).'
Write-Host '  4. Start-Apache.ps1'
Write-Host 'The legacy tree was never modified; changes made under the state root after the cutover are NOT carried back (copy them manually if needed).'
