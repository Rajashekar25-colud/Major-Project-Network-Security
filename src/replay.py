"""Offline replay session: stream a state table through the model window-by-window.

Controls: start / pause / step / reset, speed, and a time-range selector. Each
replayed window is forecast, passed through the WarningEngine (alerts) and logged.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .models.inference import Forecaster
from .models.sequences import inference_ends
from .warning.engine import WarningEngine


class ReplaySession:
    def __init__(self, states: pd.DataFrame, fc: Forecaster, config: dict[str, Any], threshold: float | None = None):
        self.states = states.reset_index(drop=True)
        self.fc = fc
        self.cfg = config
        self.thr = fc.threshold if threshold is None else float(threshold)
        self.X = fc.scale(self.states)
        ends = inference_ends(self.states, fc.L)
        p = fc.predict_batch(self.X, ends, want_state=False)
        self.ends = ends
        self.attack_prob = p["attack_prob"]
        self.stage_prob = p["stage_prob"]
        self.dec = fc.decode(self.attack_prob, self.stage_prob, self.thr)
        self.pos_of = {int(e): i for i, e in enumerate(ends)}
        self.set_range(0, len(ends) - 1)

    # ---- range / cursor ------------------------------------------------
    def set_range(self, lo: int, hi: int) -> None:
        self.lo, self.hi = int(max(lo, 0)), int(min(hi, len(self.ends) - 1))
        self.reset()

    def reset(self) -> None:
        self.cursor = self.lo - 1
        self.engine = WarningEngine(self.thr, self.cfg["warning"], self.fc.wsec, self.fc.H)
        self.events: list[dict[str, Any]] = []
        self._last_seg = None

    def set_threshold(self, thr: float) -> None:
        self.thr = float(thr)
        self.dec = self.fc.decode(self.attack_prob, self.stage_prob, self.thr)
        self.reset()

    @property
    def finished(self) -> bool:
        return self.cursor >= self.hi

    @property
    def n_total(self) -> int:
        return self.hi - self.lo + 1

    def step(self, n: int = 1) -> int:
        done = 0
        for _ in range(n):
            if self.finished:
                break
            self.cursor += 1
            k = self.cursor
            row = int(self.ends[k])
            seg = int(self.states["segment"].iat[row])
            if self._last_seg is not None and seg != self._last_seg:      # new capture segment -> new engine state
                self.engine.new_segment(k, self.states["timestamp"].iat[row])
            self._last_seg = seg
            res = self.engine.update(k, self.states["timestamp"].iat[row], float(self.dec["risk"][k]),
                                     str(self.dec["headline_stage"][k]), self.attack_prob[k].tolist(),
                                     {"row": row})
            for ev in res["events"]:
                ev["time"] = self.states["timestamp"].iat[row]
                self.events.append(ev)
            done += 1
        return done

    # ---- views -----------------------------------------------------------
    def history(self) -> pd.DataFrame:
        if self.cursor < self.lo:
            return pd.DataFrame(columns=["timestamp", "risk", "stage", "truth"])
        ks = np.arange(self.lo, self.cursor + 1)
        rows = self.ends[ks]
        return pd.DataFrame({
            "k": ks, "row": rows, "timestamp": self.states["timestamp"].to_numpy()[rows],
            "risk": self.dec["risk"][ks], "stage": self.dec["headline_stage"][ks],
            "truth": self.states["is_attack"].to_numpy()[rows], "truth_stage": self.states["stage"].to_numpy()[rows],
            "flow_count": self.states["flow_count"].to_numpy()[rows]})

    def current(self) -> dict[str, Any] | None:
        if self.cursor < self.lo:
            return None
        k = self.cursor; row = int(self.ends[k])
        return {"k": k, "row": row, "time": self.states["timestamp"].iat[row], "risk": float(self.dec["risk"][k]),
                "stage": str(self.dec["headline_stage"][k]), "probs": self.attack_prob[k], "stage_prob": self.stage_prob[k],
                "level": self.engine.level, "open_alert": self.engine.open,
                "truth": bool(self.states["is_attack"].iat[row]), "truth_stage": str(self.states["stage"].iat[row])}

    def sequence(self, k: int | None = None) -> np.ndarray:
        k = self.cursor if k is None else k
        row = int(self.ends[k])
        return self.X[row - self.fc.L + 1:row + 1]

    def forecast_table(self, k: int | None = None) -> pd.DataFrame:
        k = self.cursor if k is None else k
        rows = []
        for s in range(self.fc.H):
            sp = self.stage_prob[k, s]
            att = sp[1:] if len(sp) > 1 else sp
            top = int(att.argmax()) + 1 if len(sp) > 1 else 0
            rows.append({"step": s + 1, "seconds_ahead": (s + 1) * self.fc.wsec, "attack_probability": float(self.attack_prob[k, s]),
                         "predicted_stage": self.fc.stages[top] if self.attack_prob[k, s] >= self.thr else "Benign",
                         "most_likely_attack_stage": self.fc.stages[top], "stage_confidence": float(sp[top])})
        return pd.DataFrame(rows)

    def alerts_df(self) -> pd.DataFrame:
        return pd.DataFrame([a.as_dict() for a in self.engine.alerts]) if self.engine.alerts else pd.DataFrame()
