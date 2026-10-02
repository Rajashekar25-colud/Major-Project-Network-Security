"""AppTest harness: renders one console page (PAGE env var) with the same setup as app/console.py."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT, ROOT / "app"):
    sys.path.insert(0, str(p))
import streamlit as st

import state
import ui
from views import analyze, incidents, live, model, overview, settings, simulate

ui.inject_css()
if os.environ.get("INC_SELECT"):            # AppTest cannot click table rows: emulate selecting the first row
    from types import SimpleNamespace
    _df = st.dataframe
    st.dataframe = lambda *a, **k: (_df(*a, **{x: y for x, y in k.items() if x not in ("on_select", "selection_mode")}),
                                    SimpleNamespace(selection=SimpleNamespace(rows=[0])))[1] if k.get("on_select") else _df(*a, **k)
srcs = state.sources()
if st.session_state.get("source_label") not in srcs:
    st.session_state["source_label"] = os.environ.get("SOURCE") or state.default_source(srcs)
{"overview": overview, "live": live, "incidents": incidents, "analyze": analyze, "model": model, "settings": settings, "simulate": simulate}[os.environ["PAGE"]].render()
