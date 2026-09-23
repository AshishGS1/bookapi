import secrets

from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader

from config import settings

auth_key = APIKeyHeader(name="Auth-key", auto_error=False)


def verify_auth_key(key: str = Security(auth_key)) -> str:
    # compare_digest instead of == so we're not leaking timing info
    if not secrets.compare_digest(key, settings.auth_key):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid API key")
    return key
