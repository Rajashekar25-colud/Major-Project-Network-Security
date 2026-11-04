"""Local account store with two roles (``admin`` and ``user``).

Persistence is a plain JSON file (data/users.json) plus an append-only audit log (data/audit.jsonl).
Passwords are never stored: only a salted PBKDF2-HMAC-SHA256 hash.  Repeated failed sign-ins temporarily lock the account.

This is application-level access control for a single-machine academic deployment.  It is not a replacement for
an identity provider; use TLS and a real IdP if the dashboard is ever exposed on a network.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from ..config import ROOT

ROLES = ("admin", "user")
USERS_PATH = ROOT / "data" / "users.json"
AUDIT_PATH = ROOT / "data" / "audit.jsonl"
PBKDF2_ITERATIONS = 200_000
MAX_FAILED = 5
LOCK_MINUTES = 5
DEFAULT_ADMIN_USER = "admin"
DEFAULT_ADMIN_PASSWORD = "Admin@12345"      # first-run only; the account is forced to change it at first sign-in
_USERNAME = re.compile(r"^[A-Za-z0-9_.-]{3,32}$")
_EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_LOCK = threading.RLock()
_AUDIT_LOCK = threading.Lock()


class AuthError(ValueError):
    """A user-facing validation / authentication problem."""


# ------------------------------------------------------------------------------------------ hashing
def hash_password(password: str, salt: bytes | None = None, iterations: int = PBKDF2_ITERATIONS) -> str:
    salt = salt or secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, iterations)
    return f"pbkdf2_sha256${iterations}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, it, salt, digest = stored.split("$")
        if algo != "pbkdf2_sha256":
            return False
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), int(it))
        return hmac.compare_digest(dk.hex(), digest)
    except Exception:
        return False


def check_password_strength(password: str) -> None:
    if len(password) < 8:
        raise AuthError("Password must be at least 8 characters.")
    if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
        raise AuthError("Password must contain at least one letter and one digit.")


# ------------------------------------------------------------------------------------------ audit log
def audit(actor: str, action: str, detail: str = "", path: Path | None = None) -> None:
    p = Path(path or AUDIT_PATH)
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        rec = {"ts": datetime.now().isoformat(timespec="seconds"), "actor": actor, "action": action, "detail": detail}
        with _AUDIT_LOCK, open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
    except OSError:
        pass                                    # auditing must never break the application


def read_audit(limit: int = 200, path: Path | None = None) -> list[dict[str, Any]]:
    p = Path(path or AUDIT_PATH)
    if not p.exists():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out[::-1][:limit]


# ------------------------------------------------------------------------------------------ store
class UserStore:
    def __init__(self, path: Path | None = None, audit_path: Path | None = None):
        self.path = Path(path or USERS_PATH)
        self.audit_path = Path(audit_path or AUDIT_PATH)

    # -- persistence
    def _load(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        try:
            return json.loads(self.path.read_text(encoding="utf-8")).get("users", {})
        except Exception:
            return {}

    def _save(self, users: dict[str, dict[str, Any]]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"users": users}, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)

    @staticmethod
    def public(u: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in u.items() if k not in ("password_hash",)}

    def _log(self, actor: str, action: str, detail: str = "") -> None:
        audit(actor, action, detail, self.audit_path)

    # -- bootstrap
    def ensure_admin(self) -> bool:
        """Create the first administrator if none exists. Returns True if one was created.
        NETFORECAST_ADMIN_USER / NETFORECAST_ADMIN_PASSWORD override the defaults (then no password change is forced)."""
        with _LOCK:
            users = self._load()
            if any(u["role"] == "admin" for u in users.values()):
                return False
            name = os.environ.get("NETFORECAST_ADMIN_USER", DEFAULT_ADMIN_USER)
            env_pw = os.environ.get("NETFORECAST_ADMIN_PASSWORD")
            users[name.lower()] = self._new_user(name, env_pw or DEFAULT_ADMIN_PASSWORD, "admin", "System Administrator", "", "",
                                                 must_change=env_pw is None)
            self._save(users)
        self._log("system", "admin_created", name)
        return True

    @staticmethod
    def _new_user(username, password, role, full_name, email, organisation, must_change=False) -> dict[str, Any]:
        return {"username": username, "password_hash": hash_password(password), "role": role, "full_name": full_name, "email": email,
                "organisation": organisation, "active": True, "must_change": bool(must_change),
                "created": datetime.now().isoformat(timespec="seconds"), "last_login": None, "failed": 0, "locked_until": None}

    # -- registration / sign-in
    def create_user(self, username: str, password: str, full_name: str = "", email: str = "", organisation: str = "",
                    role: str = "user", actor: str = "self", must_change: bool = False) -> dict[str, Any]:
        username = username.strip()
        if role not in ROLES:
            raise AuthError("Unknown role.")
        if not _USERNAME.match(username):
            raise AuthError("Username must be 3-32 characters: letters, digits, '.', '_' or '-'.")
        if email and not _EMAIL.match(email.strip()):
            raise AuthError("Enter a valid e-mail address or leave it empty.")
        check_password_strength(password)
        with _LOCK:
            users = self._load()
            if username.lower() in users:
                raise AuthError("That username is already taken.")
            users[username.lower()] = self._new_user(username, password, role, full_name.strip(), email.strip(), organisation.strip(), must_change)
            self._save(users)
        self._log(actor, "user_created" if actor != "self" else "user_registered", f"{username} ({role})")
        return self.public(users[username.lower()])

    def authenticate(self, username: str, password: str) -> dict[str, Any]:
        """Return the public user record or raise AuthError. The message never reveals whether the username exists."""
        generic = AuthError("Invalid username or password.")
        key = username.strip().lower()
        with _LOCK:
            users = self._load()
            u = users.get(key)
            if u is None:
                verify_password(password, hash_password("x"))     # keep timing similar for unknown users
                self._log(username.strip() or "?", "login_failed", "unknown user")
                raise generic
            now = datetime.now()
            if u.get("locked_until") and datetime.fromisoformat(u["locked_until"]) > now:
                self._log(u["username"], "login_blocked", "account temporarily locked")
                raise AuthError(f"Too many failed attempts. Try again in {LOCK_MINUTES} minutes.")
            if not verify_password(password, u["password_hash"]):
                u["failed"] = int(u.get("failed", 0)) + 1
                if u["failed"] >= MAX_FAILED:
                    u["locked_until"] = (now + timedelta(minutes=LOCK_MINUTES)).isoformat(timespec="seconds"); u["failed"] = 0
                self._save(users)
                self._log(u["username"], "login_failed", "wrong password")
                raise generic
            if not u.get("active", True):
                self._log(u["username"], "login_blocked", "account deactivated")
                raise AuthError("This account has been deactivated. Contact an administrator.")
            u["failed"] = 0; u["locked_until"] = None; u["last_login"] = now.isoformat(timespec="seconds")
            self._save(users)
        self._log(u["username"], "login", u["role"])
        return self.public(u)

    # -- management
    def get(self, username: str) -> dict[str, Any] | None:
        u = self._load().get(username.strip().lower())
        return self.public(u) if u else None

    def list_users(self) -> list[dict[str, Any]]:
        return sorted((self.public(u) for u in self._load().values()), key=lambda u: (u["role"] != "admin", u["username"].lower()))

    def change_password(self, username: str, old: str, new: str) -> None:
        with _LOCK:
            users = self._load(); u = users.get(username.strip().lower())
            if u is None or not verify_password(old, u["password_hash"]):
                raise AuthError("Current password is incorrect.")
            check_password_strength(new)
            if verify_password(new, u["password_hash"]):
                raise AuthError("The new password must differ from the current one.")
            u["password_hash"] = hash_password(new); u["must_change"] = False
            self._save(users)
        self._log(username, "password_changed")

    def reset_password(self, username: str, new: str, actor: str) -> None:
        check_password_strength(new)
        with _LOCK:
            users = self._load(); u = users.get(username.strip().lower())
            if u is None:
                raise AuthError("No such user.")
            u["password_hash"] = hash_password(new); u["must_change"] = True; u["failed"] = 0; u["locked_until"] = None
            self._save(users)
        self._log(actor, "password_reset", username)

    def set_active(self, username: str, active: bool, actor: str) -> None:
        key = username.strip().lower()
        if key == actor.strip().lower():
            raise AuthError("You cannot deactivate your own account.")
        with _LOCK:
            users = self._load(); u = users.get(key)
            if u is None:
                raise AuthError("No such user.")
            if not active and u["role"] == "admin" and sum(1 for x in users.values() if x["role"] == "admin" and x.get("active", True)) <= 1:
                raise AuthError("At least one active administrator is required.")
            u["active"] = bool(active)
            self._save(users)
        self._log(actor, "user_activated" if active else "user_deactivated", username)
