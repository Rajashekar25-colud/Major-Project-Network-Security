"""Administrator console: system dashboard, user management and audit log. Every number comes from the stores on disk."""
from __future__ import annotations

import pandas as pd
import streamlit as st

import appstate as A
import theme as T
from src.analysis import history as H
from src.auth import read_audit
from src.auth.store import AuthError


def dashboard() -> None:
    store = A.users(); us = store.list_users(); items = H.list_history(); meta = A.metadata(); ev = A.evaluation()
    T.header("Administrator Console", "System-wide view of accounts, analyses and the trained model.", [T.chip("ADMIN", T.ACCENT)])
    if any(u["username"] == A.username() and u.get("must_change") for u in us):
        T.banner("Default password in use", "Open Profile and change the administrator password.", T.STATE_COLOR["WATCH"])
    c = st.columns(4)
    c[0].markdown(T.stat("Registered users", f"{len(us):,}", f'{sum(1 for u in us if u["role"] == "admin")} admin · {sum(1 for u in us if u.get("active", True))} active'), unsafe_allow_html=True)
    c[1].markdown(T.stat("Stored analyses", f"{len(items):,}", f'{sum(1 for m in items if m.get("labelled"))} with ground truth'), unsafe_allow_html=True)
    alerts = sum(int(m.get("alerts", 0)) for m in items)
    c[2].markdown(T.stat("Alerts raised", f"{alerts:,}", "across all stored analyses", T.STATE_COLOR["HIGH RISK"] if alerts else None), unsafe_allow_html=True)
    demo = bool(meta.get("synthetic_demo_data"))
    c[3].markdown(T.stat("Model", A.model_label(A.forecaster()), "trained on synthetic demo data" if demo else f'version {meta.get("project_version", "N/A")} · threshold {A.threshold(A.forecaster()):.0%}',
                         T.STATE_COLOR["WATCH"] if demo else T.STATE_COLOR["NORMAL"]), unsafe_allow_html=True)
    left, right = st.columns([1.5, 1], gap="large")
    with left:
        T.section("Recent analyses")
        if not items:
            st.info("No analyses have been run yet.")
        else:
            rows = [[T.esc(m.get("owner") or "N/A"), T.esc(m["filename"]), T.esc(str(m["created"]).replace("T", " ")), f'{m.get("windows", "N/A")}',
                     T.state_chip(m["state"]) if m.get("state") else "N/A", f'{int(m.get("alerts", 0))}'] for m in items[:8]]
            T.table(["Owner", "File", "Time", "Windows", "Status", "Alerts"], rows)
        if ev and "forecast" in ev:
            T.section("Held-out model quality")
            f = ev["forecast"]
            T.kv([("Precision / recall / F1", f'{T.f3(f.get("precision"))} / {T.f3(f.get("recall"))} / {T.f3(f.get("f1"))}'), ("PR-AUC", T.f3(f.get("pr_auc"))),
                  ("Held-out samples", f'{ev["n_samples"]:,}'), ("Evaluated", T.esc(str(ev.get("evaluated_at", ""))[:19].replace("T", " ")))])
    with right:
        T.section("Latest activity")
        log = read_audit(8)
        if not log:
            st.caption("No activity recorded yet.")
        else:
            T.table(["Time", "User", "Action"], [[T.esc(r["ts"][5:16].replace("T", " ")), T.esc(r["actor"]), T.esc(r["action"].replace("_", " "))] for r in log])


def users() -> None:
    store = A.users(); me = A.username()
    T.header("User Management", "Create accounts, activate or deactivate access and reset passwords.", [T.chip("ADMIN", T.ACCENT)])
    us = store.list_users()
    rows = [[f'<span class="mono">{T.esc(u["username"])}</span>', T.esc(u.get("full_name") or "N/A"), T.esc(u.get("email") or "N/A"),
             T.chip("ADMIN", T.ACCENT) if u["role"] == "admin" else T.chip("ANALYST"),
             T.chip("ACTIVE", T.STATE_COLOR["NORMAL"]) if u.get("active", True) else T.chip("DEACTIVATED", T.STATE_COLOR["HIGH RISK"]),
             T.esc(str(u.get("created", ""))[:10]), T.esc(str(u.get("last_login") or "never").replace("T", " "))] for u in us]
    T.table(["Username", "Name", "E-mail", "Role", "Status", "Created", "Last sign-in"], rows)
    c = st.columns(2, gap="large")
    with c[0]:
        T.section("Manage an account")
        names = [u["username"] for u in us if u["username"].lower() != me.lower()]
        if not names:
            st.caption("No other accounts yet.")
        else:
            who = st.selectbox("Account", names, key="adm_who")
            cur = store.get(who)
            b = st.columns(2)
            if b[0].button("Deactivate" if cur.get("active", True) else "Activate", width="stretch", key="adm_toggle"):
                try:
                    store.set_active(who, not cur.get("active", True), me); st.rerun()
                except AuthError as e:
                    st.error(str(e))
            with st.form("reset_form", clear_on_submit=True):
                npw = st.text_input("Temporary password", type="password", help="The user must change it at next sign-in.")
                if st.form_submit_button("Reset password", width="stretch"):
                    try:
                        store.reset_password(who, npw, me); st.success(f"Password reset for {who}.")
                    except AuthError as e:
                        st.error(str(e))
    with c[1]:
        T.section("Create an account")
        with st.form("create_form", clear_on_submit=True):
            un = st.text_input("Username"); fn = st.text_input("Full name"); em = st.text_input("E-mail (optional)")
            role = st.selectbox("Role", ["user", "admin"], format_func=lambda r: "Analyst" if r == "user" else "Administrator")
            pw = st.text_input("Temporary password", type="password")
            if st.form_submit_button("Create account", type="primary", width="stretch"):
                try:
                    store.create_user(un, pw, fn, em, "", role=role, actor=me, must_change=True); st.success(f"Account {un} created (password change required at first sign-in).")
                except AuthError as e:
                    st.error(str(e))


def audit_log() -> None:
    T.header("Audit Log", "Sign-ins, account changes and analyses recorded by the application.", [T.chip("ADMIN", T.ACCENT)])
    log = read_audit(500)
    if not log:
        st.info("No events have been recorded yet."); return
    df = pd.DataFrame(log).rename(columns={"ts": "Time", "actor": "User", "action": "Action", "detail": "Detail"})
    f = st.columns([1, 1, 3])
    users_f = f[0].selectbox("User", ["All"] + sorted(df["User"].unique()), key="aud_user")
    act_f = f[1].selectbox("Action", ["All"] + sorted(df["Action"].unique()), key="aud_act")
    if users_f != "All":
        df = df[df["User"] == users_f]
    if act_f != "All":
        df = df[df["Action"] == act_f]
    st.dataframe(df, hide_index=True, width="stretch", height=460)
    st.download_button("Download audit log (CSV)", df.to_csv(index=False).encode(), "audit_log.csv", "text/csv")
