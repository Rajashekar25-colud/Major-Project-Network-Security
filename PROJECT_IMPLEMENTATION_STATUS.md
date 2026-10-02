# Implementation status vs. synopsis / PPT / abstract

Legend: ✅ implemented and tested · ⚠️ implemented with a stated limitation · ⏭ future work (as in the PPT "Future Plans")

| Requirement (abstract / PPT / synopsis) | Status | Where |
|---|---|---|
| CIC-IDS2018 flow ingestion, **without** Src IP / Dst IP / Src Port | ✅ | `src/data/flows.py`, `schema.py` (only Timestamp, Dst Port, Protocol mandatory) |
| Chunked processing (no whole-file load) for any flow CSV | ✅ | `iter_flow_chunks`; test `chunked_equals_whole` |
| Time-window network state (ports, protocols, flags, packets, bytes, duration, IAT, scan patterns) | ✅ | `src/data/windows.py` (60 features) |
| Packet-level features: TTL, TCP window, retransmissions, payload, IAT | ✅ via PCAP / ⚠️ flow CSV only has TCP-window, IAT, payload | `src/data/pcap.py`; health tab reports neutralised features |
| LSTM / Transformer temporal model | ✅ LSTM + multi-head temporal self-attention | `src/models/world_model.py` |
| Multi-step forecast of future states, stage and attack probability | ✅ separate head output per future window (+ autoregressive rollout) | `world_model.py`, `inference.py` |
| MITRE ATT&CK stage mapping | ✅ configurable label→tactic table | `configs/config.yaml` |
| Explainability (SHAP or attention) | ✅ attention + gradient×input + optional exact SHAP | `src/explain/explainer.py` |
| Offline Streamlit UI: probability timeline, stages, flagged traffic, explanations | ✅ | `app/streamlit_app.py` |
| Replay mode with Start / Pause / Step / Reset / Speed / Time-range | ✅ | sidebar; `src/replay.py` |
| Centralised warning engine (persistence, cooldown, severity, dedupe) | ✅ | `src/warning/engine.py` |
| Alert history, model/data health, feature-mismatch validation | ✅ | dashboard tabs |
| Precision, Recall, F1, FPR, **Warning lead time** (early / late / missed reported) | ✅ | `src/evaluation/metrics.py` |
| Comparable forecasting baselines | ✅ logistic (sequence, per-horizon) + persistence oracle | `evaluate.py` |
| Leakage-free evaluation (per-day chronological split with gaps, train-only scaling, validation-calibrated threshold) | ✅ | `sequences.py`, `train.py` |
| Complete metadata (preprocessing version, hyperparameters, timestamps, dataset, environment) | ✅ | `models/metadata.json` |
| Tests | ✅ 29 tests (`pytest` is in requirements) | `tests/` |
| Cross-dataset testing (UNSW-NB15), graph model, live capture, defensive simulation | ⏭ | PPT "Future Plans" |
| Counterfactual network-state generation / trajectory comparison (PPT "research potential") | ⚠️ autoregressive rollout exists; counterfactual comparison is not implemented | `WorldModel.rollout` |

## Update 2 (after first real-data run)
* Validation loss rose every epoch on the real data: the per-day chronological split put different attack types in train and validation, and classes seen only in validation had no training signal. Now: train-only stage vocabulary, `blocked` split, light regularisation (dropout 0.3, input noise, lower LR).
* Web attacks (SQL injection / XSS) were dropped by the 5% attack-fraction rule; the rule is now >=3 attack flows and >=1%.
* Isolated fragments shorter than L+H+1 windows are pruned and reported (`[SEGMENTS]` line).
* `scripts/inspect_states.py` added.

## Fixes versus the previous version
* The old generated state file had broken timestamps (1970 dates) and negative durations. Timestamp parsing is now format-detected (day-first CIC format), negative values are clipped.
* Real label spellings are mapped (`Infilteration`, `Bot`, `DoS attacks-*`, `DDOS attack-*`, `Brute Force -XSS/Web`, ...).
* Forecast heads used to produce the *same* value at every horizon; each future window now has its own output.
* Threshold is calibrated on validation data (not hard-coded 0.7) and baselines are real future forecasters.
* Stage evaluation no longer reports nine-class accuracy when only a few classes exist in the hold-out.

## Limitations to state in the report
1. **I could not run the pipeline on the real 6.4 GiB dataset here.** It was developed and tested against synthetic data that reproduces the CIC-IDS2018 column schema, timestamp format, header-row and `Infinity` quirks. Run steps in README §3 to obtain your real numbers; do not quote the demo model.
2. Public datasets label the attack, not its preparation. "Forecasting" therefore means predicting an upcoming labelled attack class from preceding traffic; ATT&CK mapping is a judgement call (e.g. DoS→Impact, Infilteration→Lateral Movement) and should be reviewed.
3. The processed CSV is flow-based; packet-level claims are demonstrated through the PCAP path, which needs PCAP data for training to affect the model.
4. Some attack types occur on a single day; per-day splitting puts the later part of each day in the hold-out, but a type that occurs only briefly may still have little support. The Evaluation tab lists per-class support.
5. Patentability of the "counterfactual / rollout" idea requires a formal prior-art review (as the PPT states).
