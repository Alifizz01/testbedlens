"""Check a test-bed log against a JSON requirement spec and write a traceable report."""
import csv
import json
from pathlib import Path

import numpy as np

from .model import step_plant


def load_log(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    log = {k: np.array([float(r[k]) for r in rows]) for k in rows[0]}
    log["N1"] = 0.5 * (log["N1_A"] + log["N1_B"])
    log["N1_pred"] = predict_n1(log)
    return log


def predict_n1(log):
    """Replay the logged fuel flow through the plant model to get an expected N1."""
    state = [log["N1"][0], log["N2"][0], log["EGT"][0]]
    out, lit = [], False
    for wf, n2 in zip(log["Wf"], log["N2"]):
        out.append(state[0])
        lit = lit or n2 >= 20
        state = step_plant(state, wf, lit, starter=not lit or state[1] < 50)
    return np.array(out)


def step_info(log):
    """Step time, initial value and target of the N1 response."""
    dem = log["N1_dem"]
    idx = np.flatnonzero(np.abs(np.diff(dem)) > 1e-6)
    k = idx[0] + 1 if idx.size else 0
    return k, log["N1"][max(k - 1, 0)], dem[-1]


def response(log, band_pct):
    k, y0, yf = step_info(log)
    t, y = log["t"] - log["t"][k], log["N1"]
    span = yf - y0
    rel = (y[k:] - y0) / span
    overshoot = max(0.0, (rel.max() - 1) * 100)
    t10 = t[k:][np.argmax(rel >= 0.1)]
    t90 = t[k:][np.argmax(rel >= 0.9)] if (rel >= 0.9).any() else np.inf
    outside = np.flatnonzero(np.abs(y[k:] - yf) > band_pct / 100 * abs(span))
    if outside.size == 0:
        settle = 0.0
    elif outside[-1] == len(rel) - 1:
        settle = np.inf  # never settled inside the log
    else:
        settle = t[k:][outside[-1] + 1]
    return dict(overshoot=overshoot, rise=t90 - t10, settling=settle)


def measure(req, log):
    c, s = req["check"], req.get("signal")
    if c == "max":
        return float(log[s].max())
    if c == "rate":
        return float(np.abs(np.diff(log[s]) / np.diff(log["t"])).max())
    if c in ("overshoot", "rise", "settling"):
        return float(response(log, req.get("band_pct", 2))[c])
    if c == "channel_agreement":
        return float(np.abs(log[s + "_A"] - log[s + "_B"]).max())
    if c == "model_residual":
        return float(np.abs(log[s] - log[s + "_pred"]).max())
    raise ValueError(f"unknown check {c}")


def diagnose(log, tol):
    """Separate a sensor fault from a control-response problem using duplex channels and the model."""
    res = {ch: np.abs(log[f"N1_{ch}"] - log["N1_pred"]) for ch in "AB"}
    bad = [ch for ch in "AB" if res[ch].max() > tol]
    if not bad:
        return "No N1 sensor fault indicated: both channels agree with the model prediction."
    if len(bad) == 1:
        ch = bad[0]
        t0 = log["t"][np.argmax(res[ch] > tol)]
        return (f"Sensor fault suspected on N1 channel {ch} from t = {t0:.2f} s: it deviates from the "
                f"model prediction while the other channel follows it. Response failures in this run "
                f"are likely caused by the bad measurement, not by the control law.")
    return "Both N1 channels deviate from the model: suspect the model, fuel metering or a common-mode fault."


def plot(req, log, value, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c, s = req["check"], req.get("signal")
    fig, ax = plt.subplots(figsize=(7, 3))
    t = log["t"]
    if c in ("channel_agreement", "model_residual"):
        for name in (f"{s}_A", f"{s}_B", f"{s}_pred"):
            ax.plot(t, log[name], label=name, lw=1)
    elif c in ("overshoot", "rise", "settling"):
        ax.plot(t, log["N1"], label="N1 (mean A/B)", lw=1)
        ax.plot(t, log["N1_dem"], "--", label="N1 demand", lw=1)
    else:
        ax.plot(t, log[s], label=s, lw=1)
        if c == "max":
            ax.axhline(req["limit"], color="r", ls=":", label="limit")
    ax.set_title(f"{req['id']}: measured {value:.3g} (limit {req['limit']})", fontsize=9)
    ax.set_xlabel("t [s]")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)


def evaluate(log, spec):
    results = []
    for req in spec["requirements"]:
        v = measure(req, log)
        results.append(dict(req, value=v, passed=bool(v <= req["limit"])))
    return results


def report(log_path, spec_path, out_dir, plots=True):
    log, spec = load_log(log_path), json.loads(Path(spec_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    results = evaluate(log, spec)
    diag = diagnose(log, spec.get("residual_tol", 2.0))
    lines = [f"# Test report: {Path(log_path).stem}", "",
             f"Log: `{log_path}`  ", f"Spec: `{spec_path}`", "",
             "| Req | Requirement | Measured | Limit | Result | Plot |", "|---|---|---|---|---|---|"]
    for r in results:
        png = f"{r['id']}.png"
        if plots:
            plot(r, log, r["value"], out / png)
        lines.append(f"| {r['id']} | {r['text']} | {r['value']:.3g} | {r['limit']} | "
                     f"{'PASS' if r['passed'] else 'FAIL'} | [plot]({png}) |")
    n_fail = sum(not r["passed"] for r in results)
    lines += ["", f"**Summary:** {len(results) - n_fail} pass, {n_fail} fail.", "",
              "## Diagnosis", "", diag, ""]
    (out / "report.md").write_text("\n".join(lines))
    return results, diag
