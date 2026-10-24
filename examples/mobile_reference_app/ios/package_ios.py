"""Link an iPhoneOS SDK host and assemble an unsigned IPA on Windows.

Sideloadly or AltStore performs the Apple-account signing/install stage.  This
script deliberately requires an independently obtained iPhoneOS SDK; it does
not redistribute Apple SDK contents.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import zipfile


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sdk-root", type=Path, required=True)
    parser.add_argument("--product", type=Path, default=Path("build/mobile-reference/ios"))
    parser.add_argument("--output", type=Path, default=Path("build/mobile-reference/TuringMobile-unsigned.ipa"))
    args = parser.parse_args()
    sdk = args.sdk_root.resolve()
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
    from importlib.util import find_spec
    ziglang = find_spec("ziglang")
    if ziglang is None or ziglang.origin is None:
        raise SystemExit("the repository ziglang package is required")
    zig = Path(ziglang.origin).parent / ("zig.exe" if __import__("os").name == "nt" else "zig")
    app = product / "Payload/TuringMobile.app"
    app.mkdir(parents=True, exist_ok=True)
    executable = app / "TuringMobile"
    source = Path(__file__).with_name("main.mm")
    command = [str(zig), "c++", "-target", "arm64-apple-ios15.0", "-isysroot", str(sdk),
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
    print("Sign/install this IPA with Sideloadly or AltStore; compilation is already complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
