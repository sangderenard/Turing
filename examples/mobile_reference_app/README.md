# Turing mobile reference application

This example has one authored program, one main-output shader component, and
three shells. `program.py` is compiled through
`lower_ast_source_to_ssa`; no shell reimplements its RGB calculation.

## Build everything

From the Turing repository root:

```powershell
py -3.11 examples/mobile_reference_app/build_mobile.py
```

The pinned local toolchain is Android SDK platform 37.0, build-tools 37.0.0,
platform-tools 37.0.1, and NDK 30.0.16248370. The shader compiler and
SPIRV-Cross come from the installed Vulkan SDK.

## Android phone

The build produces
`build/mobile-reference/android/turing-mobile-debug.apk`. Enable developer
options and USB debugging, connect an ARM64 Android 7.0+ phone, accept its RSA
prompt, then run:

```powershell
powershell -ExecutionPolicy Bypass -File examples/mobile_reference_app/android/install_android.ps1
```

The script uses the repository-local `adb`, installs the signed debug APK, and
launches its `NativeActivity`. The shell requires a Vulkan-capable phone. Drag
on the screen to change green and blue; red animates from the Turing program.

## iPhone now: browser product

Build and serve the PWA:

```powershell
py -3.11 examples/mobile_reference_app/build_mobile.py --web
powershell -ExecutionPolicy Bypass -File examples/mobile_reference_app/serve_phone.ps1
```

Open the printed LAN address in Safari on the same Wi-Fi. The page invokes the
public compiler entry from `mobile_main.wasm`, then feeds its RGB result into
the page's main-output shader. Safari can add the page to the Home Screen.

The Wasm binary comes directly from the independent repository-SSA Wasm
backend. `turing-wasm-shell.js` is a generic binding over its published
invocation contract. The page is a replaceable consumer, equivalent to a C
shell: it chooses placement, feeds, scheduling, and presentation; the backend
owns no page, animation, input, or shader behavior.

## iPhone native boundary

The Windows build produces:

- `build/mobile-reference/ios/mobile_main.o`, a Mach-O ARM64 compiler object;
- the exact generated buffer ABI header;
- validated SPIR-V plus SPIRV-Cross-generated Metal 2.0 vertex and fragment
  source;
- an Objective-C++ UIKit/Metal host; and
- a separate unsigned-IPA linker/packager.

Native linking requires a legally obtained iPhoneOS SDK. Given its root:

```powershell
py -3.11 examples/mobile_reference_app/ios/package_ios.py --sdk-root C:\path\to\iPhoneOS.sdk
```

This attempts the SDK link and assembles
`build/mobile-reference/TuringMobile-unsigned.ipa`. Sideloadly or AltStore can
then perform Apple-account signing and installation. They do not replace the
SDK link. This Windows host has no iPhoneOS SDK, so the final link and physical
iPhone launch are intentionally not claimed as verified.

MoltenVK 1.4.2 is hash-pinned and staged under
`build/mobile-dependencies/MoltenVK-ios-1.4.2` for a Vulkan-over-Metal shell.
This reference host instead uses the more direct SPIR-V → MSL → Metal path;
both consume the same shader source and compiler program contract.
