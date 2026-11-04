"""Sign-in / registration (landing) and the account profile page."""
from __future__ import annotations

import streamlit as st

import appstate as A
import theme as T
from src.auth.store import DEFAULT_ADMIN_USER, AuthError


def _engine_status() -> list[tuple[bool, str]]:
    """Real readiness checks: model artifacts, a loadable forecaster, and a warning-engine configuration."""
    out = []
    ok_model = A.model_available()
    out.append((ok_model, "Model ready" if ok_model else "Model artifacts missing"))
    fc_ok = False
    if ok_model:
        try:
            A.forecaster(); fc_ok = True
        except Exception:
            fc_ok = False
    out.append((fc_ok, "Forecast engine ready" if fc_ok else "Forecast engine unavailable"))
    try:
        from src.warning.engine import WarningEngine  # noqa: F401
        w_ok = bool(A.config().get("warning"))
    except Exception:
        w_ok = False
    out.append((w_ok, "Warning engine ready" if w_ok else "Warning engine not configured"))
    return out


def login() -> None:
    st.markdown(T.HIDE_SIDEBAR, unsafe_allow_html=True)
    left, right = st.columns([1.05, 1], gap="large")
    with left:
        rows = "".join(f'<div class="sysrow"><span class="dot {"ok" if ok else "bad"}"></span>{T.esc(t)}</div>' for ok, t in _engine_status())
        st.markdown(
            '<div class="loginbrand"><div class="tag">AI Network Security</div><h2>Attack Forecasting &amp;<br>SOC Intelligence</h2>'
            '<p>Forecasts the next minute of network attack activity from traffic windows using an LSTM with temporal attention, '
            'raises early warnings, maps them to MITRE ATT&amp;CK stages and explains every prediction.</p>'
            f'{rows}<div class="pipeline">Traffic → Features → Time-window state → LSTM + attention → Multi-step forecast → '
            'Risk / warning engine → ATT&amp;CK stage → Explainability</div></div>', unsafe_allow_html=True)
    with right:
        st.markdown("&nbsp;")
        tab_in, tab_reg = st.tabs(["Sign in", "Create account"])
        with tab_in:
            _sign_in_form()
        with tab_reg:
            _register_form()


def _sign_in_form() -> None:
    portal = st.radio("Sign in as", ["Analyst", "Administrator"], horizontal=True, key="login_portal")
    with st.form("login_form", clear_on_submit=False):
        u = st.text_input("Username", key="login_user")
        p = st.text_input("Password", type="password", key="login_pass")
        go = st.form_submit_button("Sign In", type="primary", width="stretch")
    if go:
        if not u or not p:
            st.error("Enter your username and password."); return
        try:
            user = A.users().authenticate(u, p)
        except AuthError as e:
            st.error(str(e)); return
        want = "admin" if portal == "Administrator" else "user"
        if user["role"] != want:
            A.log("login_portal_mismatch", f'{user["username"]} used the {portal} portal')
            st.error("This account is an administrator account. Choose “Administrator”." if user["role"] == "admin" else "This account is not an administrator account.")
            return
        A.sign_in(user); st.rerun()
    adm = A.users().get(DEFAULT_ADMIN_USER)
    if adm and adm.get("must_change") and adm["role"] == "admin":
        st.caption(f"First run: the default administrator account “{DEFAULT_ADMIN_USER}” is active. Its password is in the README and must be changed at first sign-in.")


def _register_form() -> None:
    with st.form("register_form", clear_on_submit=False):
        name = st.text_input("Full name")
        email = st.text_input("E-mail (optional)")
        org = st.text_input("Organisation (optional)")
        ru = st.text_input("Username", key="reg_user", help="3–32 characters: letters, digits, '.', '_' or '-'")
        p1 = st.text_input("Password", type="password", key="reg_p1", help="At least 8 characters with a letter and a digit")
        p2 = st.text_input("Confirm password", type="password", key="reg_p2")
        go = st.form_submit_button("Create account", width="stretch")
    if go:
        if p1 != p2:
            st.error("The passwords do not match."); return
        try:
            A.users().create_user(ru, p1, name, email, org, role="user")
        except AuthError as e:
            st.error(str(e)); return
        st.success("Account created. Open the “Sign in” tab to continue.")


# ---------------------------------------------------------------------------------------------- profile
def profile() -> None:
    u = A.current_user(); store = A.users()
    T.header("Profile", "Your account, access role and password.", [T.chip(u["role"].upper(), T.ACCENT if u["role"] == "admin" else None)])
    if u.get("must_change"):
        T.banner("Password change required", "This account is using a temporary or default password. Set a new one to continue.", T.STATE_COLOR["WATCH"])
    c = st.columns([1.2, 1], gap="large")
    with c[0]:
        T.section("Account")
        T.kv([("Username", f'<span class="mono">{T.esc(u["username"])}</span>'), ("Full name", T.esc(u.get("full_name") or "N/A")), ("E-mail", T.esc(u.get("email") or "N/A")),
              ("Organisation", T.esc(u.get("organisation") or "N/A")), ("Role", "Administrator" if u["role"] == "admin" else "Analyst"),
              ("Created", T.esc(str(u.get("created", "N/A")).replace("T", " "))), ("Last sign-in", T.esc(str(u.get("last_login") or "N/A").replace("T", " ")))])
    with c[1]:
        T.section("Change password")
        with st.form("pw_form", clear_on_submit=True):
            old = st.text_input("Current password", type="password")
            n1 = st.text_input("New password", type="password")
            n2 = st.text_input("Confirm new password", type="password")
            go = st.form_submit_button("Update password", type="primary", width="stretch")
        if go:
            if n1 != n2:
                st.error("The new passwords do not match.")
            else:
                try:
                    store.change_password(u["username"], old, n1)
                except AuthError as e:
                    st.error(str(e))
                else:
                    A.refresh_user(); st.success("Password updated."); st.rerun()
    if u["role"] == "user":
        from src.analysis import history as H
        mine = H.list_history(owner=u["username"])
        T.section("Your activity")
        T.kv([("Saved analyses", f"{len(mine):,}"), ("Alerts across your analyses", f'{sum(int(m.get("alerts", 0)) for m in mine):,}')])
