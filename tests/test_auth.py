"""Account store: hashing, validation, roles, lock-out, audit trail. Pure Python (no torch / streamlit)."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.auth.store import AuthError, UserStore, hash_password, read_audit, verify_password  # noqa: E402


@pytest.fixture()
def store(tmp_path):
    s = UserStore(tmp_path / "users.json", tmp_path / "audit.jsonl")
    s.ensure_admin()
    return s


def test_hash_roundtrip_and_salt():
    h1, h2 = hash_password("Secret123"), hash_password("Secret123")
    assert h1 != h2 and verify_password("Secret123", h1) and not verify_password("secret123", h1)
    assert not verify_password("x", "garbage")


def test_default_admin_must_change_password(store):
    assert store.ensure_admin() is False                        # idempotent
    a = store.authenticate("ADMIN", "Admin@12345")              # case-insensitive username
    assert a["role"] == "admin" and a["must_change"] and "password_hash" not in a
    store.change_password("admin", "Admin@12345", "NewPass123")
    assert store.get("admin")["must_change"] is False
    with pytest.raises(AuthError):
        store.authenticate("admin", "Admin@12345")


def test_register_validation(store):
    store.create_user("alice", "Passw0rdX", "Alice", "a@b.co")
    for args in (("al", "Passw0rdX"), ("alice", "Passw0rdX"), ("bob", "short1"), ("bob", "onlyletters"), ("bob", "Passw0rdX", "", "not-an-email")):
        with pytest.raises(AuthError):
            store.create_user(*args)
    assert store.get("alice")["role"] == "user"


def test_lockout_and_deactivation(store):
    store.create_user("carol", "Passw0rdX")
    for _ in range(5):
        with pytest.raises(AuthError):
            store.authenticate("carol", "wrong")
    with pytest.raises(AuthError, match="Too many"):
        store.authenticate("carol", "Passw0rdX")
    store.create_user("dave", "Passw0rdX")
    store.set_active("dave", False, "admin")
    with pytest.raises(AuthError, match="deactivated"):
        store.authenticate("dave", "Passw0rdX")


def test_admin_safeguards_and_audit(store):
    with pytest.raises(AuthError):
        store.set_active("admin", False, "admin")               # cannot deactivate yourself / last admin
    store.create_user("erin", "Passw0rdX")
    store.reset_password("erin", "Another123", "admin")
    assert store.authenticate("erin", "Another123")["must_change"]
    actions = [r["action"] for r in read_audit(path=tmp_audit(store))]
    assert {"admin_created", "user_registered", "password_reset", "login"} <= set(actions)


def tmp_audit(store):
    return store.audit_path
