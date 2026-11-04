# Diagram package

Every diagram describes the **implemented** system (verified against the code), not a planned one. Sources are text files so they can be edited and re-rendered with `python scripts/render_diagrams.py`.

| # | Diagram | Source | Image | Corresponds to |
|---|---------|--------|-------|----------------|
| 1 | System architecture | `01_system_architecture.dot` | `.png` `.svg` | `app/` (Streamlit), `src/auth`, `src/analysis`, `src/data`, `src/features`, `src/models`, `src/replay.py`, `src/warning`, `src/explain`, `src/evaluation`, files under `models/`, `data/`, `configs/` |
| 2 | Use case | `02_use_case.dot` | `.png` `.svg` | Role-based pages in `app/main.py` (Analyst vs Administrator) |
| 3 | Class | `03_class_diagram.mmd` | `.png` `.svg` | Real classes: `UserStore`, `TrafficReport`, `Check`, `WindowAccumulator`, `Preprocessor`, `WorldModel`, `TransformerWorldModel`, `Forecaster`, `ReplaySession`, `WarningEngine`, `Alert`, `AnalysisResult`; plus modules `history`, `explainer`, `metrics/validation`, `integrity` |
| 4 | Sequence | `04_sequence_diagram.mmd` | `.png` `.svg` | New Analysis flow: `new_analysis.py` -> `ingest.inspect_traffic` -> `workflow.run_analysis` -> `Forecaster` -> `WarningEngine` -> `history.save_analysis` -> dashboard |
| 5 | Activity | `05_activity_diagram.dot` | `.png` `.svg` | Same flow with the validation and threshold decisions |
| 6 | ER (logical) | `06_er_diagram.mmd` | `.png` `.svg` | Files actually persisted (see below) |
| 7 | Level-1 DFD | `07_level1_dfd.dot` | `.png` `.svg` | Processes 1.0-6.0, data store D1, three external entities |

`Network_Attack_Forecasting_Diagrams.pdf` contains all seven figures, one per page, for inserting into the report. `actor.png` is the stick figure used by the use-case diagram.

## Accuracy notes (read before presenting)

* **No relational database.** The application persists to files: `data/users.json` (accounts), `data/audit.jsonl` (audit log), `data/history/<id>.json` + `<id>.states.csv.gz` (analyses), and `models/*` (trained model, metadata, evaluation). The ER diagram is therefore a **logical** data model of those files, not a PostgreSQL schema. (`legacy/aegis_console` has its own SQLite file; it is not part of the main Streamlit app and is not drawn.)
* **Forecast, Alert and Explanation are not persisted entities.** They are recomputed deterministically from the stored network-state windows and the fixed trained model when an analysis is opened. Only the alert count, peak risk and state are stored in the analysis record. They are shown as classes (class diagram) but not as ER tables.
* **`NetworkState` / `TrafficWindow`** are rows of the `states` DataFrame produced by `WindowAccumulator`; there is no separate class, so none is drawn. `AnalysisHistory`, `Explainer`, `EvaluationMetrics` and `ModelIntegrity` are Python modules of functions and are marked `<<module>>`.
* **Explanation and ATT&CK mapping are lazy.** They are computed when the Explain / ATT&CK page is opened (SHAP only on request), which is why the sequence diagram shows them in an `opt` block after the dashboard, not before storage.
* **MITRE ATT&CK** is mapped at tactic (stage) level from `configs/config.yaml`; technique IDs are not modelled.
* **Roles.** Analyst: run and review analyses, own history, profile. Administrator: user management, all analyses, audit log, model health (incl. alert threshold), evaluation; can also run analyses.

## Actors, processes and stores

* **Actors**: SOC Analyst / Security Operator, System Administrator, Network Traffic Source (the uploaded capture), ML Forecasting System (secondary actor).
* **DFD processes**: 1.0 Traffic Ingestion, 2.0 Feature Processing, 3.0 Attack Forecasting, 4.0 Risk & Warning Analysis, 5.0 Explanation & ATT&CK Analysis, 6.0 Dashboard & Reporting.
* **Data store D1**: `models/`, `data/history/`, `data/users.json`, `data/audit.jsonl`.
