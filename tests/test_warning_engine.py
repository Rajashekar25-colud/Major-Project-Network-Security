from src.warning.engine import WarningEngine

W = dict(watch_ratio=0.7, critical_margin=0.5, persistence_windows=2, clear_ratio=0.8, clear_windows=3, cooldown_windows=5)


def run(risks, stages=None, thr=0.5, w=W):
    e = WarningEngine(thr, w, 10, 6)
    for i, r in enumerate(risks):
        e.update(i, i, r, (stages or ["Impact"] * len(risks))[i])
    return e


def test_severity_levels():
    e = WarningEngine(0.5, W, 10, 6)
    assert [e.level_of(r) for r in (0.1, 0.4, 0.6, 0.9)] == ["OK", "WATCH", "WARNING", "CRITICAL"]


def test_persistence_blocks_single_spike():
    assert len(run([0.1, 0.9, 0.1, 0.1]).alerts) == 0
    assert len(run([0.1, 0.9, 0.9, 0.9]).alerts) == 1


def test_deduplication_and_hysteresis():
    e = run([0.9] * 3 + [0.45] * 2 + [0.9] * 3)       # dips never clear (need 3 windows below 0.4)
    assert len(e.alerts) == 1 and e.alerts[0].status == "OPEN"


def test_close_and_cooldown():
    r = [0.9, 0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.9]       # re-attack right after closing, same stage
    e = run(r)
    assert e.alerts[0].status == "CLOSED" and len(e.alerts) == 1      # suppressed by cooldown
    e2 = run(r + [0.1] * 3 + [0.9] * 3 + [0.9] * 3, stages=["Impact"] * 18)
    assert len(e2.alerts) >= 1


def test_stage_progression_and_escalation():
    e = run([0.6, 0.6, 0.6, 0.95], ["Reconnaissance", "Reconnaissance", "Initial Access", "Initial Access"])
    a = e.alerts[0]
    assert a.stages == ["Reconnaissance", "Initial Access"] and a.peak_severity == "CRITICAL"


def test_new_segment_closes_but_keeps_history():
    e = run([0.9, 0.9, 0.9]); e.new_segment(3, 3)
    assert e.open is None and e.alerts[0].status == "CLOSED" and len(e.alerts) == 1
