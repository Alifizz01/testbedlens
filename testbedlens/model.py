"""Simplified two-spool turbofan transient model and synthetic test-bed log generator.

This is a teaching model, not an engine model. Spool speeds are in percent,
EGT in degC, fuel flow Wf is normalised 0..1. Dynamics are first-order lags:

    N2' = (N2_ss(Wf) - N2) / TAU_N2      N2_ss = 60 + 45 * Wf   (lit)
    N1' = (N1_ss(N2) - N1) / TAU_N1      N1_ss = 25 + 1.6 * (N2 - 60)
    EGT' = (EGT_ss - EGT) / TAU_EGT      EGT_ss = 380 + 4 * (N2 - 60) + 600 * excess fuel

"Excess fuel" is Wf above the steady-state fuel needed for the current N2, so
EGT peaks during accelerations as on a real engine (qualitatively only).
A PI controller tracks an N1 demand from throttle, with Wf min/max and rate limits.
"""
import csv
import numpy as np

DT = 0.02
TAU_N2, TAU_N1, TAU_EGT = 1.0, 1.5, 2.0
WF_RATE = 0.4          # max |dWf/dt| per second (accel/decel schedule stand-in)
KP, KI = 0.020, 0.010  # nominal PI gains on N1 error in %
COLS = ["t", "TLA", "N1_dem", "Wf", "N1_A", "N1_B", "N2", "EGT"]


def n1_demand(tla):
    return 25 + 0.72 * tla


def step_plant(state, wf, lit, starter, dt=DT):
    """Advance [N1, N2, EGT] by one step. Shared by simulator and analysis predictor."""
    n1, n2, egt = state
    if lit:
        n2_ss = 60 + 45 * wf
        wf_needed = max(0.0, (n2 - 60) / 45)
        egt_ss = 380 + 4 * (n2 - 60) + 600 * max(0.0, wf - wf_needed)
    else:
        n2_ss = 25.0 if starter else 0.0
        egt_ss = 20.0
    n1_ss = max(0.0, 25 + 1.6 * (n2 - 60))
    n2 += dt * (n2_ss - n2) / TAU_N2
    n1 += dt * (n1_ss - n1) / TAU_N1
    egt += dt * (egt_ss - egt) / TAU_EGT
    return [n1, n2, egt]


def simulate(tla_fn, duration, start=False, kp=KP, ki=KI, fault=None, seed=0):
    """Closed-loop run. fault = dict(kind=drift|stuck|noise, t0, t1, size) on channel N1_A."""
    rng = np.random.default_rng(seed)
    if start:
        state, lit, wf, integ = [0.0, 0.0, 20.0], False, 0.0, 0.0
    else:
        wf0 = (n1_demand(tla_fn(0)) - 25) / 1.6 / 45
        state = [n1_demand(tla_fn(0)), 60 + 45 * wf0, 380 + 4 * 45 * wf0]
        lit, wf, integ = True, wf0, wf0
    rows = []
    for i in range(int(duration / DT) + 1):
        t = i * DT
        tla = tla_fn(t)
        n1, n2, egt = state
        a = n1 + rng.normal(0, 0.1)
        b = n1 + rng.normal(0, 0.1)
        if fault and fault["t0"] <= t <= fault["t1"]:
            k, s = fault["kind"], fault["size"]
            if k == "drift":
                a += s * (t - fault["t0"])
            elif k == "stuck":
                a = fault.setdefault("_held", a)
            elif k == "noise":
                a += rng.normal(0, s)
        n1_meas = 0.5 * (a + b)  # ponytail: plain average, a real FADEC uses channel selection logic
        dem = n1_demand(tla)
        if not lit and n2 >= 20:
            lit = True
        if lit:
            err = dem - n1_meas
            integ = float(np.clip(integ + ki * err * DT, 0, 1))
            wf_cmd = np.clip(integ + kp * err, 0, 1)
            wf = float(np.clip(wf_cmd, wf - WF_RATE * DT, wf + WF_RATE * DT))
            if wf != wf_cmd:
                integ = wf - kp * err  # anti-windup: hold integrator while a limiter is active
        rows.append([t, tla, dem, wf, a, b, n2, egt])
        state = step_plant(state, wf, lit, starter=start and n2 < 50)
    return rows


def write_csv(rows, path):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        w.writerows([[f"{v:.4f}" for v in r] for r in rows])


def step(t0, a, b):
    return lambda t: a if t < t0 else b


SCENARIOS = {
    "accel_step": dict(tla_fn=step(2, 20, 80), duration=20),
    "decel_step": dict(tla_fn=step(2, 80, 20), duration=20),
    "start": dict(tla_fn=lambda t: 0, duration=30, start=True),
    "fault_drift": dict(tla_fn=step(2, 20, 80), duration=20, fault=dict(kind="drift", t0=8, t1=20, size=0.6)),
    "fault_stuck": dict(tla_fn=step(2, 20, 80), duration=20, fault=dict(kind="stuck", t0=3, t1=20, size=0)),
    "fault_noise": dict(tla_fn=step(2, 20, 80), duration=20, fault=dict(kind="noise", t0=10, t1=13, size=4.0)),
    "mistuned_gain": dict(tla_fn=step(2, 20, 80), duration=20, kp=0.004, ki=0.06)  # deliberately mistuned,
}


def generate(name, path):
    cfg = dict(SCENARIOS[name])
    if cfg.get("fault"):
        cfg["fault"] = dict(cfg["fault"])
    write_csv(simulate(**cfg), path)
