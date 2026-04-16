import base64
import hashlib
import hmac
import secrets
from datetime import datetime, timedelta, timezone


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=64,
    )
    return (
        "scrypt$"
        + base64.b64encode(salt).decode("ascii")
        + "$"
        + base64.b64encode(digest).decode("ascii")
    )


def verify_password(password: str, stored_hash: str) -> bool:
    try:
        algo, b64salt, b64digest = stored_hash.split("$", 2)
    except ValueError:
        return False
    if algo != "scrypt":
        return False
    salt = base64.b64decode(b64salt.encode("ascii"))
    expected = base64.b64decode(b64digest.encode("ascii"))
    actual = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=2**14,
        r=8,
        p=1,
        dklen=64,
    )
    return hmac.compare_digest(actual, expected)


def new_session_token() -> str:
    return secrets.token_urlsafe(48)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def session_expiry(hours: int = 24) -> datetime:
    return utc_now() + timedelta(hours=hours)
