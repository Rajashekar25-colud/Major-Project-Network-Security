"""Smoke test for the application layer: imports every UI module and renders each page headlessly with Streamlit's AppTest.
Run:  python scripts/check_app_imports.py      (needs streamlit + torch; uses the real trained model in models/)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "app"))
from streamlit.testing.v1 import AppTest  # noqa: E402

at = AppTest.from_file(str(ROOT / "app" / "main.py"), default_timeout=120).run()
assert not at.exception, at.exception
print("PASS  app/main.py renders the upload screen;", len(at.button), "buttons")
for page in ("history", "settings", "evaluation"):
    at.session_state["page"] = page; at.run()
    assert not at.exception, (page, at.exception); print("PASS ", page)
print("OK")
