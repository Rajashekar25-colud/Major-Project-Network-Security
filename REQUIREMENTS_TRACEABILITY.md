# Requirements traceability - PPT, synopsis and abstract vs. the implementation

Status: ✅ implemented and tested · ⚠️ implemented with a stated limitation · ⏭ not implemented (listed as future work in the PPT)

## Abstract / synopsis - system behaviour
| Requirement | Status | Evidence |
|---|---|---|
| Flow-level features: ports, protocols, packet counts, bytes, duration, inter-arrival times, TCP flags | ✅ | `src/data/windows.py` (60 features per window) |
| Packet-level features: TTL, TCP window size, retransmissions, payload, inter-arrival | ⚠️ | Computed from PCAP (`src/data/pcap.py`). CIC-IDS2018 CSVs only contain TCP initial-window bytes; TTL and retransmissions are empty there, so a model trained on CSVs does not use them (shown as "telemetry the model never saw"). Training on PCAP-derived states is needed to use them |
| Port-scanning patterns | ✅ | `scan_score`, distinct-port count, failed/small-flow ratios |
| Time-window network state over multiple windows | ✅ | 10 s windows, 12 windows of history, 6-window (60 s) forecast; all configurable |
| LSTM **or** Transformer | ✅ | `model.type: lstm | transformer` in `configs/config.yaml`; `scripts/compare_models.py` trains both and compares (PPT "Model upgrade") |
| Learn state transitions and predict future states | ✅ | state head + autoregressive roll-out (`WorldModel.rollout`) |
| Map predictions to MITRE ATT&CK stages | ✅ | label→tactic table in the config; tactic IDs and response playbooks in `src/product/knowledge.py`. Reconnaissance/Exfiltration have no samples in CSE-CIC-IDS2018, so the model cannot learn them from it |
| SHAP or attention explanations | ✅ | attention, gradient×input and exact SHAP (Incidents → Advanced) |
| Offline Streamlit interface: infiltration-probability timeline, predicted stages, flagged traffic patterns, explanations | ✅ | `app/console.py` - Overview / Live Monitor / Incidents ("Flagged traffic patterns" tables) |
| Fully offline | ✅ | no network calls; SQLite + local files only |

## Synopsis - objectives
| # | Objective | Status | Where |
|---|---|---|---|
| 1 | Collect and preprocess network traffic data | ✅ | chunked CSV reader, header-row / Infinity / negative-value handling |
| 2 | Extract important features | ✅ | `windows.py`, train-only scaling `features/preprocess.py` |
| 3 | Analyse traffic over different time intervals | ✅ | `window.seconds` configurable; per-day segments and gap handling |
| 4 | Develop an LSTM or Transformer model | ✅ | both available |
| 5 | Forecast future attack stages | ✅ | per-horizon stage + attack-probability heads |
| 6 | Map to ATT&CK | ✅ | see above |
| 7 | Early warnings | ⚠️ | Alert engine with severity, persistence, cool-down. **Measured early-warning performance depends on the data**: on the real CIC-IDS2018 hold-out most episodes were detected at onset rather than minutes ahead (see Model & Learning → Detection quality). Sensitivity presets trade false alarms for earlier warning |
| 8 | Explain important features | ✅ | plain-language drivers + charts |
| 9 | Evaluate with Precision, Recall, F1, FPR, Warning Lead Time | ✅ | `src/evaluation/`; lead-time reports early / late / missed episodes and false alerts per hour. Accuracy is also reported |
| 10 | Streamlit dashboard | ✅ | `app/console.py` |

## Synopsis - setup, datasets, results
| Requirement | Status | Evidence |
|---|---|---|
| Batch 32, 20-50 epochs, learning rate 0.001, Adam, dropout 0.2-0.5 | ✅ | defaults in `configs/config.yaml` (batch 32, up to 50 epochs with early stopping, lr 0.001, Adam, dropout 0.3); test `test_synopsis_hyperparameters_are_the_defaults` |
| Precision / Recall / F1 / FPR / Warning lead time | ✅ | `models/evaluation.json`, Model & Learning page |
| CIC-IDS2018 | ✅ | processed CSVs, chunked; run by you on the real data |
| UNSW-NB15 | ⚠️ | `profiles.unsw_nb15` handles the header-less raw files, second/millisecond units, `attack_cat` labels and non-TCP/UDP protocols; tested only on **generated UNSW-format data**. Run it on the real raw files, then `scripts/cross_dataset_eval.py` |
| Reduced computational complexity / improved scalability | ✅ | constant-memory chunked ingestion; `scripts/benchmark.py` measures throughput (synthetic: ~87k flow rows/s ingestion, ~7.9k windows/s forecasting on one CPU core) |
| Training/testing split without leakage | ✅ | blocked per-day split, train-only scaling and vocabulary, validation-calibrated threshold |

## PPT - expected outcomes
| Outcome | Status | Evidence |
|---|---|---|
| Early warning | ⚠️ | see objective 7 |
| Multi-step forecast | ✅ | 6 future windows, per-horizon metrics |
| Explainable result | ✅ | |
| Useful evaluation | ✅ | |
| Simple dashboard | ✅ | |
| Research & patent potential: counterfactual network-state generation, future roll-out, trajectory comparison | ✅ | `src/product/counterfactual.py`, **Defence Simulation** page; test `test_counterfactual_changes_generated_states`. Patentability itself requires a formal prior-art review (as the PPT says) |

## PPT - future plans
| Plan | Status | Notes |
|---|---|---|
| More datasets (UNSW-NB15, others) | ⚠️ | supported and unit-tested on generated data; needs a run on the real files |
| Graph-based network model | ⏭ | not implemented |
| Real-time monitoring on live telemetry | ⏭ | the Live Monitor replays stored traffic; no live capture |
| Defensive simulation (no action / block IP / isolate host) | ✅ | Defence Simulation page. Scenario effects are **assumptions** (editable in `configs/product.yaml`), not validated causal effects |
| Model upgrade: LSTM vs Transformer | ✅ | `scripts/compare_models.py` (needs your real data for meaningful numbers) |
| Deployment of the offline Streamlit system | ⚠️ | runs locally with one command, product settings and optional password; no installer or container image |

## What must be run on your machine
1. `python scripts/train.py --states data\states\cicids2018_states.csv` - retrain with the synopsis hyper-parameters.
2. `python scripts/compare_models.py --states data\states\cicids2018_states.csv --epochs 20` - LSTM vs Transformer table.
3. `python scripts/benchmark.py` - throughput on your hardware.
4. (optional) UNSW-NB15: `prepare_states.py` on the raw files, then `cross_dataset_eval.py`.
5. `python scripts/self_check.py` - overall health check.
