"""Link the native host with an iPhoneOS SDK and assemble an unsigned IPA.

On macOS the SDK and Apple linker are resolved through ``xcrun``.  Other hosts
must provide a legally obtained SDK root and use the repository's existing
ziglang cross-toolchain.  Signing and installation remain a separate stage.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile


def capture(command: list[str]) -> str:
    return subprocess.run(
        command, check=True, capture_output=True, text=True,
    ).stdout.strip()


def resolve_sdk(explicit: Path | None) -> Path:
    if explicit is not None:
        return explicit.resolve()
    if sys.platform != "darwin":
        raise SystemExit("--sdk-root is required outside macOS")
    xcrun = shutil.which("xcrun")
    if not xcrun:
        raise SystemExit("xcrun is required to locate the iPhoneOS SDK")
    return Path(capture([xcrun, "--sdk", "iphoneos", "--show-sdk-path"]))


def compiler_command() -> list[str]:
    if sys.platform == "darwin":
        xcrun = shutil.which("xcrun")
        if not xcrun:
            raise SystemExit("xcrun is required for the Apple platform link")
        return [xcrun, "--sdk", "iphoneos", "clang++"]
    from importlib.util import find_spec
    ziglang = find_spec("ziglang")
    if ziglang is None or ziglang.origin is None:
        raise SystemExit("the repository ziglang package is required")
    return [
        str(Path(ziglang.origin).parent / ("zig.exe" if os.name == "nt" else "zig")),
        "c++",
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-root", type=Path)
    parser.add_argument("--product", type=Path, default=Path("build/mobile-reference/ios"))
    parser.add_argument("--output", type=Path, default=Path("build/mobile-reference/TuringMobile-unsigned.ipa"))
    args = parser.parse_args()
    sdk = resolve_sdk(args.sdk_root)
    if not (sdk / "System/Library/Frameworks/UIKit.framework").exists():
        raise SystemExit(f"not an iPhoneOS SDK: {sdk}")
    product = args.product.resolve()
    required = [product / name for name in (
        "mobile_main.o", "generated/turing_program_abi.h", "fullscreen.vert.metal",
        "main_output.frag.metal",
    )]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("run build_mobile.py --ios first; missing: " + ", ".join(missing))
    app = product / "Payload/TuringMobile.app"
    app.mkdir(parents=True, exist_ok=True)
    executable = app / "TuringMobile"
    source = Path(__file__).with_name("main.mm")
    command = [*compiler_command(), "-target", "arm64-apple-ios15.0", "-isysroot", str(sdk),
               "-fobjc-arc", "-std=c++20", "-I", str(product), str(source),
               str(product / "mobile_main.o"), "-framework", "UIKit", "-framework", "Metal",
               "-framework", "MetalKit", "-framework", "QuartzCore", "-o", str(executable)]
    print("[ios-package] " + " ".join(command), flush=True)
    subprocess.run(command, check=True)
    shutil.copy2(Path(__file__).with_name("Info.plist"), app / "Info.plist")
    shutil.copy2(product / "fullscreen.vert.metal", app / "fullscreen.vert.metal")
    shutil.copy2(product / "main_output.frag.metal", app / "main_output.frag.metal")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in (product / "Payload").rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(product).as_posix())
    print(f"Unsigned IPA: {args.output.resolve()}")
    print("Signing and device installation are separate from this compiler/link product.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
