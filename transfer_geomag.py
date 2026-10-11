"""
TRANSFER TEST - same associator AND-gate as the crypto result, new domain.

  crypto                              -> geomagnetic
  8 Binance pairs, hourly             -> 8 OMNI solar-wind channels, hourly
  hourly log returns                  -> hourly changes (log-change for positive quantities)
  event: BTC vol crosses 95th pct     -> event: Kp reaches 5 (G1 storm) onset
  baseline: BTC realised-vol z-score  -> baseline: solar-wind electric field z-score
Lags, windows, smoothing, threshold tuning, nulls: unchanged.

Setup: pip install numpy pandas requests
Run:   python transfer_geomag.py
"""
import os, json
import numpy as np
import pandas as pd
import requests

OUT = "transfer_geomag"
YEARS = range(2010, 2026)
HOLDOUT = "2023-01-01"
H, W, SMOOTH = 24, 720, 24     # horizon, trailing window, smoothing (hours)
ALERT_GAP, EVENT_GAP = 24, 72
LAGS = (0, 1, 3)
N_NULL = 1000
rng = np.random.default_rng(0)
os.makedirs(OUT, exist_ok=True)

# OMNI2 hourly: name -> (0-based column, fill value, log-change?)
CHANNELS = {
    "B_mag": (8, 999.9, True),     # slot 0 = real part: drops out of the associator
    "Bx":    (12, 999.9, False),
    "By":    (15, 999.9, False),
    "Bz":    (16, 999.9, False),
    "T":     (22, 9999999.0, True),
    "N":     (23, 999.9, True),
    "V":     (24, 9999.0, True),
    "P":     (28, 99.99, True),
}
EXTRA = {"E": (35, 999.99), "Kp": (38, 99.0)}   # E = -V*Bz (mV/m), Kp stored x10

# ------------------------------------------------------------------ data
def load_year(y):
    path = os.path.join(OUT, f"omni2_{y}.dat")
    if not os.path.exists(path):
        url = f"https://spdf.gsfc.nasa.gov/pub/data/omni/low_res_omni/omni2_{y}.dat"
        r = requests.get(url, timeout=120)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"  downloaded {y}")
    a = np.loadtxt(path)
    idx = (pd.to_datetime(a[:, 0].astype(int).astype(str), format="%Y", utc=True)
           + pd.to_timedelta(a[:, 1] - 1, unit="D")
           + pd.to_timedelta(a[:, 2], unit="h"))
    cols = {k: (v[0], v[1]) for k, v in CHANNELS.items()} | EXTRA
    out = {}
    for name, (i, fill) in cols.items():
        v = a[:, i].astype(float)
        v[v >= fill * 0.999] = np.nan
        out[name] = v
    return pd.DataFrame(out, index=idx)

# ------------------------------------------------------------ octonions
FANO = [(1, 2, 4), (2, 3, 5), (3, 4, 6), (4, 5, 7), (5, 6, 1), (6, 7, 2), (7, 1, 3)]

def _unit_rules(C):
    C[0, 0, 0] = 1
    for i in range(1, 8):
        C[0, i, i] = C[i, 0, i] = 1
        C[i, i, 0] = -1

def table(triples, signs=None):
    C = np.zeros((8, 8, 8)); _unit_rules(C)
    for n, (i, j, k) in enumerate(triples):
        s = 1 if signs is None else signs[n]
        for a, b, c in [(i, j, k), (j, k, i), (k, i, j)]:
            C[a, b, c], C[b, a, c] = s, -s
    return C

def random_antisym():
    C = np.zeros((8, 8, 8)); _unit_rules(C)
    for i in range(1, 8):
        for j in range(i + 1, 8):
            k = rng.choice([x for x in range(1, 8) if x not in (i, j)])
            s = rng.choice([-1, 1])
            C[i, j, k], C[j, i, k] = s, -s
    return C

def mul(C, x, y):
    return np.einsum("ijk,ti,tj->tk", C, x, y)

def assoc_norm(C, a, b, c):
    A = mul(C, mul(C, a, b), c) - mul(C, a, mul(C, b, c))
    return np.linalg.norm(A, axis=1)

# ------------------------------------------------------------- helpers
def zfeat(x):
    sm = x.rolling(SMOOTH).mean()
    return (sm - sm.rolling(W).mean()) / sm.rolling(W).std()

def thin(idx, gap):
    keep, last = [], -10**9
    for i in idx:
        if i - last >= gap:
            keep.append(i); last = i
    return np.array(keep, dtype=int)

def score(mask, seg):
    k = thin(np.flatnonzero(mask), ALERT_GAP)
    y, yb = seg["y"].values, seg["y_back"].values
    on = np.flatnonzero(seg["onset"].values)
    if len(k) == 0:
        return dict(n=0, hits=0, lift=np.nan, recall=0.0, lift_back=np.nan)
    caught = sum(1 for o in on if ((k >= o - H) & (k < o)).any())
    return dict(n=int(len(k)), hits=int(y[k].sum()),
                lift=float(y[k].mean() / y.mean()),
                recall=caught / max(len(on), 1),
                lift_back=float(yb[k].mean() / yb.mean()))

# --------------------------------------------------------------- build
print("Loading OMNI data...")
d = pd.concat([load_year(y) for y in YEARS]).sort_index()
d = d[~d.index.duplicated()].asfreq("1h")
d = d.interpolate(limit=6, limit_area="inside").dropna()

print("Sanity check (if these look wrong, a column index is off):")
print(f"  median V {d.V.median():.0f} km/s (expect ~350-450) | "
      f"median N {d.N.median():.1f} /cc (expect ~3-7) | max Kp {d.Kp.max()/10:.1f} (expect 7-9)")

X = pd.DataFrame({c: (np.log(d[c].clip(lower=1e-3)) if lg else d[c]).diff()
                  for c, (_, _, lg) in CHANNELS.items()}, index=d.index)

kp = d["Kp"].values
above = kp >= 50
cross = np.flatnonzero(above & ~np.r_[False, above[:-1]])
onset = np.zeros(len(d), bool); onset[thin(cross, EVENT_GAP)] = True
base_raw = d["E"].rolling(3).mean()

n = len(X); t = np.arange(n)
cs = np.r_[0, np.cumsum(onset)]
df = pd.DataFrame(index=X.index)
df["base_z"] = (base_raw - base_raw.rolling(W).mean()) / base_raw.rolling(W).std()
Z = (X - X.rolling(W).mean()) / X.rolling(W).std()
a, b, c = (Z.shift(L).values for L in LAGS)
signs = rng.choice([-1, 1], 7)
tables = {"fano": table(FANO), "fano_randsign": table(FANO, signs),
          "rand_antisym": random_antisym()}
for k, C in tables.items():
    df[k] = zfeat(pd.Series(assoc_norm(C, a, b, c), index=X.index))
amp = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
df["amp_only"] = zfeat(pd.Series(amp, index=X.index))
df["onset"] = onset
df["y"] = (cs[np.minimum(t + H + 1, n)] - cs[t + 1]) > 0       # storm onset in (t, t+H]
df["y_back"] = (cs[t] - cs[np.maximum(t - H, 0)]) > 0           # storm onset in [t-H, t)
df = df[df.notna().all(axis=1).values & (t < n - H)]

cut = pd.Timestamp(HOLDOUT, tz="UTC")
ins, hol = df[df.index < cut], df[df.index >= cut]

# ------------------------------------------- tune on in-sample only
tb = ins.base_z.quantile(0.90)

def tune(name):
    best = None
    for q in np.arange(0.50, 0.96, 0.05):
        ta = ins[name].quantile(q)
        s = score(((ins.base_z > tb) & (ins[name] > ta)).values, ins)
        if s["n"] >= 30 and (best is None or s["lift"] > best[1]["lift"]):
            best = (ta, s)
    return best

tuned = {k: tune(k) for k in ["fano", "fano_randsign", "rand_antisym", "amp_only"]}
if tuned["fano"] is None:
    raise SystemExit("No associator threshold gave >=30 in-sample alerts - gate cannot be tuned.")
ta_f, s_in = tuned["fano"]
tb2 = min((ins.base_z.quantile(q) for q in np.linspace(0.90, 0.999, 200)),
          key=lambda v: abs(score((ins.base_z > v).values, ins)["n"] - s_in["n"]))

# ------------------------------------------------- holdout results
def gate(name, ta):
    return ((hol.base_z > tb) & (hol[name] > ta)).values

rows = {"baseline only": score((hol.base_z > tb).values, hol),
        "AND gate (fano)": score(gate("fano", ta_f), hol),
        "baseline, matched rate": score((hol.base_z > tb2).values, hol)}
for k in ["fano_randsign", "rand_antisym", "amp_only"]:
    rows[f"AND gate ({k})"] = score(gate(k, tuned[k][0]), hol) if tuned[k] else None

obs = rows["AND gate (fano)"]
y = hol.y.values; base = y.mean()

kb = thin(np.flatnonzero((hol.base_z > tb).values), ALERT_GAP)
m = min(obs["n"], len(kb))
null_veto = np.array([y[rng.choice(kb, m, replace=False)].mean() / base
                      for _ in range(N_NULL)])

fa, bz = hol.fano.values, (hol.base_z > tb).values
null_shift = []
for _ in range(N_NULL // 2):
    s = score(bz & (np.roll(fa, rng.integers(W, len(fa) - W)) > ta_f), hol)
    if s["n"] > 0:
        null_shift.append(s["lift"])
null_shift = np.array(null_shift)

# ------------------------------------------------------------- report
print(f"\nIn-sample AND gate (fano): lift {s_in['lift']:.2f}, alerts {s_in['n']}")
print(f"Holdout {HOLDOUT} -> end | base rate {base:.2%} | storm onsets {int(hol.onset.sum())}")
print(f"\n{'variant':30s} {'alerts':>6s} {'hits':>5s} {'lift':>6s} {'recall':>7s} {'lift_back':>9s}")
for k, s in rows.items():
    if s is None:
        print(f"{k:30s}  (no threshold with >=30 alerts in-sample)"); continue
    print(f"{k:30s} {s['n']:6d} {s['hits']:5d} {s['lift']:6.2f} "
          f"{s['recall']:7.1%} {s['lift_back']:9.2f}")

bl = rows["baseline only"]["lift"]
mr = rows["baseline, matched rate"]["lift"]
p_veto = float((null_veto >= obs["lift"]).mean())
p_shift = float((null_shift >= obs["lift"]).mean())
checks = {
    "beats baseline alone":       obs["lift"] > bl,
    "beats matched-rate baseline": obs["lift"] > mr,
    "beats random veto (p<0.05)":  p_veto < 0.05,
    "timing matters (p<0.05)":     p_shift < 0.05,
}
print("\nPRE-REGISTERED CRITERIA")
for k, v in checks.items():
    print(f"  {'PASS' if v else 'FAIL'}  {k}")
print(f"        random-veto p = {p_veto:.3f}, shift p = {p_shift:.3f}")
print(f"\nImprovement over baseline: {obs['lift'] / bl - 1:+.0%}   (crypto was +48%)")
for k in ["fano_randsign", "rand_antisym", "amp_only"]:
    s = rows[f"AND gate ({k})"]
    if s:
        print(f"  {k:14s} retains {s['lift'] / obs['lift']:.0%} of fano lift")
print(f"\nVERDICT: {'METHOD TRANSFERS' if all(checks.values()) else 'DOES NOT TRANSFER (on these criteria)'}")
if obs["hits"] < 10:
    print(f"WARNING: only {obs['hits']} hits on holdout - too few to trust either way.")

json.dump({"in_sample": s_in, "holdout": rows, "checks": checks,
           "p_veto": p_veto, "p_shift": p_shift, "randsign_signs": signs.tolist()},
          open(os.path.join(OUT, "results.json"), "w"), indent=2, default=float)
print(f"Saved {OUT}/results.json")
