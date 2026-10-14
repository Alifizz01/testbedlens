import pytest

from testbedlens.analyze import diagnose, evaluate, load_log, report
from testbedlens.model import SCENARIOS, generate

SPEC = "specs/step.json"


@pytest.fixture(scope="module")
def logs(tmp_path_factory):
    d = tmp_path_factory.mktemp("logs")
    out = {}
    for name in SCENARIOS:
        generate(name, d / f"{name}.csv")
        out[name] = load_log(d / f"{name}.csv")
    return out


def failed(log, spec=SPEC):
    import json
    from pathlib import Path
    return {r["id"] for r in evaluate(log, json.loads(Path(spec).read_text())) if not r["passed"]}


@pytest.mark.parametrize("name", ["accel_step", "decel_step"])
def test_clean_steps_pass_everything(logs, name):
    assert failed(logs[name]) == set()
    assert diagnose(logs[name], 2.0).startswith("No N1 sensor fault")


def test_clean_start_passes(logs):
    assert failed(logs["start"], "specs/start.json") == set()


@pytest.mark.parametrize("name", ["fault_drift", "fault_stuck", "fault_noise"])
def test_injected_sensor_faults_are_flagged_on_channel_a(logs, name):
    assert {"REQ-SNS-001", "REQ-SNS-002"} <= failed(logs[name])
    assert "channel A" in diagnose(logs[name], 2.0)


def test_drift_detected_after_onset_not_before(logs):
    msg = diagnose(logs["fault_drift"], 2.0)
    t = float(msg.split("t = ")[1].split(" s")[0])
    assert 8.0 < t < 13.0  # drift starts at 8 s at 0.6 %/s


def test_mistuned_gain_is_control_problem_not_sensor(logs):
    f = failed(logs["mistuned_gain"])
    assert "REQ-RSP-001" in f
    assert not f & {"REQ-SNS-001", "REQ-SNS-002"}
    assert diagnose(logs["mistuned_gain"], 2.0).startswith("No N1 sensor fault")


def test_model_replay_matches_clean_log(logs):
    import numpy as np
    log = logs["accel_step"]
    assert np.abs(log["N1_B"] - log["N1_pred"]).max() < 0.5


def test_report_written_with_plots(tmp_path):
    generate("mistuned_gain", tmp_path / "m.csv")
    report(tmp_path / "m.csv", SPEC, tmp_path / "rep")
    md = (tmp_path / "rep" / "report.md").read_text()
    assert "| REQ-RSP-001 |" in md and "FAIL" in md
    assert (tmp_path / "rep" / "REQ-RSP-001.png").stat().st_size > 1000
