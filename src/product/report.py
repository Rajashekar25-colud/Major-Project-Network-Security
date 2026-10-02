"""Self-contained HTML security report (no external resources; prints cleanly to PDF)."""
from __future__ import annotations

import html
from typing import Any

import numpy as np
import pandas as pd

from .config import SEV_COLOR
from .knowledge import playbook


def _spark(values: np.ndarray, thr: float, w: int = 900, h: int = 120) -> str:
    if len(values) < 2:
        return ""
    step = max(len(values) // 600, 1)
    v = np.asarray(values)[::step]
    xs = np.linspace(8, w - 8, len(v)); ys = h - 8 - np.clip(v, 0, 1) * (h - 16)
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    ty = h - 8 - thr * (h - 16)
    return (f'<svg viewBox="0 0 {w} {h}" width="100%" height="{h}"><rect width="{w}" height="{h}" fill="#f8fafc" rx="8"/>'
            f'<line x1="8" x2="{w - 8}" y1="{ty:.1f}" y2="{ty:.1f}" stroke="#f97316" stroke-dasharray="5 4"/>'
            f'<polyline points="{pts}" fill="none" stroke="#0891b2" stroke-width="1.6"/></svg>')


def build_report(brand: dict[str, Any], source: str, summary: dict[str, Any], incidents: pd.DataFrame,
                 risk: np.ndarray, thr: float, model_version: str) -> str:
    e = html.escape
    rows = ""
    for r in incidents.sort_values("opened_time").head(200).itertuples():
        c = SEV_COLOR.get(r.severity, "#64748b")
        rows += (f"<tr><td>{e(str(r.id))}</td><td>{e(str(r.opened_time))[:19]}</td><td><span style='color:{c};font-weight:700'>{e(r.severity)}</span></td>"
                 f"<td>{e(r.stage)}</td><td>{e(str(r.progression))}</td><td>{r.peak_risk:.0%}</td><td>{r.eta_seconds:.0f}s</td></tr>")
    stages = incidents["stage"].unique() if len(incidents) else []
    guide = "".join(f"<h3>{e(s)} <small>({playbook(s)['id']})</small></h3><p>{e(playbook(s)['meaning'])}</p><ul>"
                    + "".join(f"<li>{e(a)}</li>" for a in playbook(s)["actions"]) + "</ul>" for s in stages)
    sev = ", ".join(f"{v} {k}" for k, v in summary.get("by_severity", {}).items()) or "none"
    org = f" · {e(brand['organisation'])}" if brand.get("organisation") else ""
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{e(brand['product_name'])} - Threat report</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;color:#0f172a;max-width:980px;margin:32px auto;padding:0 20px}}
h1{{margin:0}}h2{{border-bottom:2px solid #e2e8f0;padding-bottom:6px;margin-top:34px}}.sub{{color:#64748b}}
.kpis{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}}.k{{flex:1;min-width:150px;border:1px solid #e2e8f0;border-radius:10px;padding:12px}}
.k b{{display:block;font-size:22px}}.k span{{color:#64748b;font-size:12px;text-transform:uppercase;letter-spacing:.05em}}
table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:7px 8px;border-bottom:1px solid #e2e8f0;text-align:left}}th{{background:#f1f5f9}}
small{{color:#64748b;font-weight:400}}@media print{{body{{margin:0}}}}</style></head><body>
<h1>{e(brand['product_name'])}</h1><div class="sub">Network threat forecast report{org}</div>
<p class="sub">Source: <b>{e(source)}</b> · Period {e(str(summary['start']))[:19]} → {e(str(summary['end']))[:19]} · Model {e(model_version)} · Generated {pd.Timestamp.now():%Y-%m-%d %H:%M}</p>
<div class="kpis"><div class="k"><span>Incidents</span><b>{summary['incidents']}</b></div><div class="k"><span>Severity</span><b style="font-size:15px">{e(sev)}</b></div>
<div class="k"><span>Peak risk</span><b>{summary['peak_risk']:.0%}</b></div><div class="k"><span>Traffic analysed</span><b>{summary['hours']:.1f} h</b></div>
<div class="k"><span>Top tactic</span><b style="font-size:15px">{e(str(summary['top_tactic'] or '-'))}</b></div></div>
<h2>Forecast risk over time</h2>{_spark(risk, thr)}<p class="sub">Line = forecast probability of attack activity in the next minute; dashed = alert threshold ({thr:.0%}).</p>
<h2>Incidents</h2><table><tr><th>ID</th><th>Opened</th><th>Severity</th><th>Predicted tactic</th><th>Progression</th><th>Peak risk</th><th>Expected in</th></tr>{rows or '<tr><td colspan=7>No incidents.</td></tr>'}</table>
<h2>Recommended response</h2>{guide or '<p>No action required.</p>'}
<p class="sub" style="margin-top:30px">Forecasts are probabilistic. Tactic names follow MITRE ATT&amp;CK. Validate alerts against your own telemetry before taking disruptive action.</p></body></html>"""
