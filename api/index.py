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
        # Read matched path from Vercel header if present
        matched_path = request.headers.get("x-matched-path")
        if matched_path and not matched_path.startswith("/api/index"):
            request.scope["path"] = matched_path
        else:
            path = request.scope.get("path", "")
            if path in ("/api/index.py", "/api/index", "/api"):
                request.scope["path"] = "/"
            elif path.startswith("/api/index.py/"):
                request.scope["path"] = path.replace("/api/index.py", "", 1)
            elif path.startswith("/api/index/"):
                request.scope["path"] = path.replace("/api/index", "", 1)
        return await call_next(request)

app.add_middleware(VercelPathCorrectionMiddleware)
