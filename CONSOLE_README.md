# Aegis Forecast console

```powershell
python -m streamlit run app/console.py
```

| Page | What it is for |
|---|---|
| Overview | Threat posture, risk timeline, incidents by tactic, one-click HTML report |
| Live Monitor | Stream traffic window by window: live risk, 60-second forecast, likely tactic, why, recommended response |
| Incidents | Triage queue (New / Investigating / Resolved / False positive), notes, playbooks, exports |
| Analyze Traffic | Upload a PCAP/PCAPNG or flow CSV and get incidents |
| Model & Learning | Detection quality, traffic drift, **learn from new labelled data** (candidate model, comparison, approve / discard), versions and rollback |
| Settings | Sensitivity presets (Low noise / Balanced / Early warning), advanced alert behaviour |

Rename the product, add an organisation name or switch on password protection in `configs/product.yaml`.
Incidents, notes and the audit trail are stored locally in `data/app.db`; settings in `data/settings.json`.
Nothing leaves the machine.
