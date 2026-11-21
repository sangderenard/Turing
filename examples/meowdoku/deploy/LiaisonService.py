"""Windows service entry for the account-isolated outgoing MCP liaison.

Install only with Setup-LiaisonService.ps1. The service deliberately receives
no OAuth data on its command line and never binds beyond loopback.
"""
from pathlib import Path
import sys

import win32service
import win32serviceutil


ROOT = Path("C:/dev/Powershell")
TURING = ROOT / "turing"
sys.path[:0] = [str(TURING / "examples"), str(ROOT / "speaktome")]

from meowdoku.liaison import LiaisonServer, connection_service  # noqa: E402


class MeowdokuLiaison(win32serviceutil.ServiceFramework):
    _svc_name_ = "MeowdokuLiaison"
    _svc_display_name_ = "Meowdoku Suno Liaison"
    _svc_description_ = "Account-isolated outgoing MCP connection and OAuth credential vault"

    def __init__(self, args):
        super().__init__(args)
        self.server = None

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        if self.server is not None:
            self.server.shutdown()

    def SvcDoRun(self):
        state = ROOT / "srv" / "state"
        service = connection_service(state / "run" / "mcp_connections.json",
                                     state / "secrets" / "liaison_vault")
        try:
            with LiaisonServer(state / "secrets" / "liaison-ipc.key", service) as server:
                self.server = server
                server.serve_forever(poll_interval=0.2)
        finally:
            self.server = None
            service.close()


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "install-from-file":
        # Keep the service-account password out of process arguments and logs.
        password = Path(sys.argv[2]).read_text(encoding="utf-8").strip()
        win32serviceutil.InstallService(
            win32serviceutil.GetServiceClassString(MeowdokuLiaison),
            MeowdokuLiaison._svc_name_, MeowdokuLiaison._svc_display_name_,
            startType=win32service.SERVICE_AUTO_START,
            userName=r".\svc-liaison", password=password,
            description=MeowdokuLiaison._svc_description_)
    else:
        win32serviceutil.HandleCommandLine(MeowdokuLiaison)
