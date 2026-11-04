# Legacy / archived UIs

Earlier interfaces, kept only for reference. They are **not** part of the "New Analysis" workflow.

* `soc_dashboard_v2.py` - previous single-page replay dashboard.
* `aegis_console/` - the "Aegis Forecast" console. It includes incident triage (SQLite), defence what-if simulation and
  **"learn from new labelled data" (model fine-tuning)** - features the main app deliberately does not expose.

Run from the project root: `python -m streamlit run legacy/aegis_console/console.py`
Do not use it if you need a guarantee that uploaded traffic can never change the model.
