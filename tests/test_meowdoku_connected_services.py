"""Connected-services navigation and read-only account status contract."""
from pathlib import Path
import shutil
import subprocess
import unittest
from test_meowdoku_studio_web_exports import declared_exports

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "examples/meowdoku/web"

class ConnectedServicesTests(unittest.TestCase):
    def test_navigation_and_exports(self):
        self.assertTrue({"connected-services.html", "connected-services.js"} <= declared_exports())
        for name in ("keys.html", "site-nav.js"):
            self.assertIn("/connected-services.html", (WEB / name).read_text(encoding="utf-8"))
        page = (WEB / "connected-services.html").read_text(encoding="utf-8")
        self.assertIn("MCP services", page)
        self.assertIn('/keys.html#paper-box', page)
        self.assertIn('https://suno.com/mcp', page)
        self.assertIn('suno-connect', page)
        self.assertNotIn('<input', page)
        self.assertNotIn('onclick=', page)

    @unittest.skipUnless(shutil.which("node"), "Node required for isolated JS behavior tests")
    def test_status_and_repeated_interrupted_reads(self):
        result = subprocess.run([shutil.which("node"), str(ROOT / "tests/connected_services_check.cjs"),
                                 str(WEB / "connected-services.js")], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("PASS", result.stdout)

if __name__ == "__main__":
    unittest.main()
