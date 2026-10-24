param(
    [string]$Apk = "$PSScriptRoot\..\..\..\build\mobile-reference\android\turing-mobile-debug.apk"
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path "$PSScriptRoot\..\..\..").Path
$Adb = Join-Path $Repo "build\mobile-dependencies\android-sdk\platform-tools\adb.exe"
if (-not (Test-Path -LiteralPath $Adb)) { throw "Android platform-tools are missing. Run build_mobile.py --android first." }
if (-not (Test-Path -LiteralPath $Apk)) { throw "APK not found: $Apk" }
& $Adb devices
& $Adb install -r $Apk
& $Adb shell am start -n org.turingcompiler.mobile/android.app.NativeActivity
