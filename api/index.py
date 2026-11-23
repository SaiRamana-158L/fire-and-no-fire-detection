import sys
from pathlib import Path
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

# Add project root to sys.path so app module and dependencies are resolved
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app import app

class VercelPathCorrectionMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        path = request.scope.get("path", "")
        if path.startswith("/api/index.py"):
            subpath = path[len("/api/index.py"):]
            request.scope["path"] = "/" if not subpath else subpath
        elif path.startswith("/api/index"):
            subpath = path[len("/api/index"):]
            request.scope["path"] = "/" if not subpath else subpath
        return await call_next(request)

app.add_middleware(VercelPathCorrectionMiddleware)
