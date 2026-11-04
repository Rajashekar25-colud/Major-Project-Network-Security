# AI-Based Network Attack Forecasting from Network Traffic Data

MLRITM · Project batch MP-C04 · Guide: Dr. Y. Madhusekhar

Traffic → feature extraction → **time-window network state** → **LSTM + attention world model** →
**K-step forecast** → **MITRE ATT&CK stage** → **warning engine** → **SHAP / attention explanation** → **offline Streamlit SOC dashboard**.

Everything is real computation: nothing on the dashboard is hard-coded. Each replayed window is forecast by the trained model,
scored by the warning engine and logged as it is replayed.

## 1. Install (Windows PowerShell)

```powershell
cd Major-Project-Network-Security
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -c "import pandas, numpy, sklearn, torch, shap, streamlit; print('All packages OK')"
```

## 2. Run the dashboard immediately (demo model, synthetic data)

```powershell
python -m streamlit run app/main.py
```
The repo ships a small demo model trained on **synthetic** CIC-format data (the dashboard shows a yellow banner). It is only there so the UI works
before you process the real dataset. Never quote its metrics.

## 3. Real CSE-CIC-IDS2018 run (your 6.4 GiB processed CSVs)

```powershell
# (a) stream the CSVs into network states (chunked, ~200k rows at a time, low RAM)
python scripts/prepare_states.py --data "D:\CSE-CIC-IDS2018\Processed Traffic Data for ML Algorithms"

# (b) train + calibrate threshold + evaluate on the per-day chronological hold-out
python scripts/train.py --states data/states/cicids2018_states.csv

#   or both steps in one command:
python scripts/train.py --data "D:\CSE-CIC-IDS2018\Processed Traffic Data for ML Algorithms"

# (c) optional: re-run the evaluation only
python scripts/evaluate.py --states data/states/cicids2018_states.csv

# (d) dashboard - pick "cicids2018_states.csv" in the sidebar
python -m streamlit run app/main.py
```
Typical CPU cost: state building is I/O bound (a few minutes for all 10 days); training uses at most `max_train_samples_per_epoch` sequences per epoch (config) so each epoch takes minutes, not hours.
Reduce `model.hidden_size` / `epochs` in `configs/config.yaml` if needed.

The CSV reader needs only **Timestamp, Dst Port, Protocol** (+ Label for training). `Src IP`, `Dst IP`, `Src Port` are optional, so the standard processed files and the
files that include IPs both work. Embedded repeated header rows, `Infinity`/NaN values and negative durations are handled.

### 3b. Check the data before trusting results
```powershell
python scripts/inspect_states.py --states data\states\cicids2018_states.csv
```
Shows per-day windows/segments, which ATT&CK stages exist, and how many forecast samples of each stage land in train / val / test.

**Split strategy.** The default `blocked` split cuts every capture day into blocks assigned to train/val/test, so every attack type is visible to training.
Test blocks sit next to train blocks of the same campaign, so results measure *within-campaign* generalisation, not unseen campaigns - say this in the report.
The published CIC processed files for several days are cut at 1,048,575 rows (Excel limit), so those days cover only part of the day.

## 4. Packet-level data (TTL, TCP window, retransmissions, payload, inter-arrival)

```powershell
python scripts/pcap_to_states.py --pcap data/sample/sample_traffic.pcap
```
or upload a `.pcap` in the dashboard (sidebar → *Upload file*, or *Use bundled sample PCAP*). Classic `.pcap` is parsed with pure Python; `.pcapng` needs `pip install scapy`.
A model trained on flow CSVs has no TTL/retransmission signal; the **Model & data health** tab tells you which features are neutralised. To use packet-level features in the model itself,
train on a state table that includes PCAP-derived windows.

## 4b. New Analysis workflow (the application)

```powershell
python -m streamlit run app/main.py
```
**+ New Analysis** -> upload a CSV / PCAP / PCAPNG -> *Traffic ready* (real validation + statistics) -> **Begin Analysis**
(real pipeline progress) -> Overview / Traffic / Forecast / Alerts / Explain / ATT&CK / Forecast Validation / Evaluation. History and Settings are always available.

* The uploaded file is **inference data only**: the model weights and scaler are never trained, fitted or written. The Settings page shows a model fingerprint (SHA-256) and each analysis re-checks it.
* Forecast / Explain / ATT&CK all follow the **selected forecast step** (+10 s ... +K s). Everything is computed by the trained model: no placeholder values.
* Alerts come from the project's `WarningEngine` (threshold + persistence + hysteresis + cooldown), numbered 1..n (no invented IDs).
* *Forecast Validation* needs ground-truth labels in the uploaded file; otherwise the app says so. *Evaluation* shows the saved held-out artifact and is clearly separate from the current file.
* Offline replay is labelled **OFFLINE REPLAY**. Live capture is not part of this application.
* Earlier UIs (SOC dashboard, Aegis console with incident triage and model fine-tuning) were moved to `legacy/`.

## 5. Legacy dashboard (replay mode)  *(archived: `legacy/soc_dashboard_v2.py`)*

Sidebar: source, threshold, capture-day selector, **time-range slider**, **speed** (windows/s), **Start / Pause / Step / +10 / Reset / Jump**.
Tabs: **Live SOC view** (status banner, KPIs, infiltration-probability timeline, K-step forecast bars, ATT&CK stage distribution, flagged traffic),
**Explanation** (gradient×input attribution, temporal attention, optional SHAP, autoregressive world-model rollout), **Alert history**, **Model & data health**
(feature-mismatch validation, metadata), **Evaluation**, **Methodology**.

## 6. Tests

```powershell
python -m pytest -q
```
29 tests: ingestion (CIC schema without IPs, chunked == whole-file, day-first timestamps, header rows), PCAP features, leakage-free splits, warning engine (persistence, hysteresis,
cooldown, dedupe, escalation), lead-time semantics (early / late / missed / false alert), threshold calibration, training artifacts/metadata, replay controls, feature-mismatch validation.

## 7. Layout

```
configs/config.yaml        all windowing / model / threshold / warning / label-mapping settings
src/data/                  flows.py (chunked CSV) · windows.py (state builder) · pcap.py · schema.py · synthetic.py
src/features/preprocess.py train-only scaling, active-feature mask, mismatch validation
src/models/                world_model.py · sequences.py (splits) · train.py · inference.py
src/warning/engine.py      centralised alert engine
src/evaluation/            metrics.py (lead time, onset, stage) · evaluate.py (+ baselines)
src/explain/explainer.py   attention, gradient×input, SHAP
src/replay.py              replay session used by the dashboard
scripts/                   prepare_states · train · evaluate · pcap_to_states · make_demo
app/main.py                Network Forecast AI (New Analysis workflow); app/screens/ = pages
src/analysis/              ingest (validation) · workflow (forecast/alerts/explain views) · validation · history · integrity
legacy/                    archived earlier UIs (not used by the new workflow)
PROJECT_IMPLEMENTATION_STATUS.md   requirement-by-requirement status and honest limitations
```

## 8. Reading the results honestly

* Stage accuracy is computed **only over stages that occur in the hold-out**; unsupported classes are listed, not scored.
* Headline numbers: use the **onset** metrics (benign now → attack within the horizon) and **lead time**; "forecast" metrics also include windows where the attack is already underway.
* The baselines are *forecasting* baselines on identical inputs/targets/splits (logistic regression, persistence oracle).
* Dataset labels describe the attack itself, not its preparation. ATT&CK mapping is a configurable table you should review.


## Sign-in, roles and documentation diagrams

The Streamlit app (`python -m streamlit run app/main.py`) now opens on a sign-in / registration page.

| Role | Can do |
|------|--------|
| **Analyst** (self-registers) | New Analysis, Overview, Traffic, Forecast, Alerts, Explain, MITRE ATT&CK, Forecast Validation, own History, Profile |
| **Administrator** | Dashboard, Users (create / deactivate / reset password), All Analyses, Audit Log, Model Health (incl. alert threshold), Evaluation, plus everything an analyst can do |

* First run creates the administrator `admin` with the temporary password `Admin@12345`; you are forced to change it at first sign-in. Set `NETFORECAST_ADMIN_USER` / `NETFORECAST_ADMIN_PASSWORD` before the first start to choose your own.
* Accounts live in `data/users.json` (PBKDF2-SHA256 salted hashes, 5 failed attempts lock an account for 5 minutes); sign-ins and account changes are written to `data/audit.jsonl`. This is application-level access control for a single-machine deployment, not an identity provider. Sessions end when the browser tab is closed or reloaded.
* Saved analyses record their owner; analysts see only their own, administrators see all. Analyses saved before this change have no owner and are visible to administrators only.
* UML / DFD / ER / architecture diagrams are in `docs/diagrams/` (sources, PNG, SVG, a combined PDF) with a report-ready write-up in `docs/REPORT_DIAGRAMS.md`. Re-render with `python scripts/render_diagrams.py`.
