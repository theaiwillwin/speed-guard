"""
Minimal audit of the crypto associator AND-gate.
Rebuilt from the method description, not the original code.
Setup:  pip install numpy pandas requests
Run:    python assoc_audit.py     (first run downloads ~6 yrs of hourly bars, then caches)
"""
import os, time, json
import numpy as np
import pandas as pd
import requests

OUT = "assoc_audit"
PAIRS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT",
         "ADAUSDT", "SOLUSDT", "DOGEUSDT", "LTCUSDT"]
START, END, HOLDOUT = "2020-09-01", "2026-10-01", "2025-01-01"
H = 24            # forecast horizon (hours)
W = 720           # trailing window (30 days)
ALERT_GAP = 24    # min hours between counted alerts
EVENT_GAP = 72    # min hours between counted events (stops threshold flicker)
LAGS = (0, 1, 3)  # octonions a, b, c = returns at t, t-1, t-3
N_NULL = 1000
rng = np.random.default_rng(0)
os.makedirs(OUT, exist_ok=True)

# ------------------------------------------------------------------ data
def fetch(sym):
    path = os.path.join(OUT, f"{sym}_1h.csv")
    if os.path.exists(path):
        s = pd.read_csv(path, index_col=0)["close"]
        s.index = pd.to_datetime(s.index, utc=True)
        return s
    url = "https://data-api.binance.vision/api/v3/klines"
    t = int(pd.Timestamp(START, tz="UTC").timestamp() * 1000)
    end = int(pd.Timestamp(END, tz="UTC").timestamp() * 1000)
    rows = []
    while t < end:
        r = requests.get(url, params={"symbol": sym, "interval": "1h",
                                      "startTime": t, "limit": 1000}, timeout=30)
        r.raise_for_status()
        k = r.json()
        if not k:
            break
        rows += k
        t = k[-1][0] + 3_600_000
        time.sleep(0.05)
    s = pd.Series([float(x[4]) for x in rows],
                  index=pd.to_datetime([x[0] for x in rows], unit="ms", utc=True),
                  name="close")
    s = s[~s.index.duplicated()]
    s.to_frame().to_csv(path)
    print(f"  {sym}: {len(s)} bars")
    return s

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
def zfeat(x):  # smooth 24h, then z-score vs trailing 30 days (past only)
    sm = x.rolling(24).mean()
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
print("Loading data...")
px = pd.concat({s: fetch(s) for s in PAIRS}, axis=1).sort_index()
px = px[px.index < pd.Timestamp(END, tz="UTC")]
px = px.asfreq("1h").ffill(limit=6).dropna()
r = np.log(px).diff().dropna()
n = len(r); t = np.arange(n)

# event: BTC 24h realised vol first crosses its trailing 30d 95th percentile
rv = r["BTCUSDT"].rolling(24).std()
thr = rv.shift(1).rolling(W).quantile(0.95)
above = (rv > thr).values & thr.notna().values
cross = np.flatnonzero(above & ~np.r_[False, above[:-1]])
onset = np.zeros(n, bool); onset[thin(cross, EVENT_GAP)] = True

cs = np.r_[0, np.cumsum(onset)]
y_fwd = (cs[np.minimum(t + H + 1, n)] - cs[t + 1]) > 0     # event in (t, t+H]
y_back = (cs[t] - cs[np.maximum(t - H, 0)]) > 0           # event in [t-H, t)

# features (all causal: use data up to and including t)
vol_z = (rv - rv.rolling(W).mean()) / rv.rolling(W).std()
Z = (r - r.rolling(W).mean()) / r.rolling(W).std()
a, b, c = (Z.shift(L).values for L in LAGS)
signs = rng.choice([-1, 1], 7)
variants = {"fano": table(FANO),
            "fano_randsign": table(FANO, signs),
            "rand_antisym": random_antisym()}
feat = {k: zfeat(pd.Series(assoc_norm(C, a, b, c), index=r.index))
        for k, C in variants.items()}
amp = np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1) * np.linalg.norm(c, axis=1)
feat["amp_only"] = zfeat(pd.Series(amp, index=r.index))

df = pd.DataFrame({"vol_z": vol_z, **feat}, index=r.index)
df["onset"], df["y"], df["y_back"] = onset, y_fwd, y_back
df = df[df.notna().all(axis=1).values & (t < n - H)]
cut = pd.Timestamp(HOLDOUT, tz="UTC")
ins, hol = df[df.index < cut], df[df.index >= cut]

# ------------------------------------------- tune on in-sample only
tv = ins["vol_z"].quantile(0.90)

def tune(name):
    best = None
    for q in np.arange(0.50, 0.96, 0.05):
        ta = ins[name].quantile(q)
        s = score(((ins.vol_z > tv) & (ins[name] > ta)).values, ins)
        if s["n"] >= 30 and (best is None or s["lift"] > best[1]["lift"]):
            best = (ta, s)
    return best

ta_f, s_in = tune("fano")
target = s_in["n"]
tv2 = min((ins.vol_z.quantile(q) for q in np.linspace(0.90, 0.999, 200)),
          key=lambda v: abs(score((ins.vol_z > v).values, ins)["n"] - target))

# ------------------------------------------------- holdout results
rows = {
    "baseline: vol only": score((hol.vol_z > tv).values, hol),
    "AND gate (fano)": score(((hol.vol_z > tv) & (hol.fano > ta_f)).values, hol),
    "vol only, matched rate": score((hol.vol_z > tv2).values, hol),
}
for name in ["fano_randsign", "rand_antisym", "amp_only"]:
    tb = tune(name)
    rows[f"AND gate ({name})"] = (score(((hol.vol_z > tv) & (hol[name] > tb[0])).values, hol)
                                  if tb else None)

obs = rows["AND gate (fano)"]
y = hol.y.values; base = y.mean()

# null 1: random veto - keep the same number of baseline alerts at random
kb = thin(np.flatnonzero((hol.vol_z > tv).values), ALERT_GAP)
m = min(obs["n"], len(kb))
null_veto = np.array([y[rng.choice(kb, m, replace=False)].mean() / base
                      for _ in range(N_NULL)])

# null 2: time-shift the associator by >= 30 days (keeps shape, breaks timing)
fa, vz = hol.fano.values, (hol.vol_z > tv).values
null_shift = []
for _ in range(N_NULL // 2):
    s = score(vz & (np.roll(fa, rng.integers(W, len(fa) - W)) > ta_f), hol)
    if s["n"] > 0:
        null_shift.append(s["lift"])
null_shift = np.array(null_shift)

# ------------------------------------------------------------- report
print(f"\nIn-sample AND gate (fano): lift {s_in['lift']:.2f}, alerts {s_in['n']} "
      f"(original reported ~6.35)")
print(f"Holdout {HOLDOUT} -> end | base rate {base:.2%} | events {int(hol.onset.sum())}")
print(f"\n{'variant':30s} {'alerts':>6s} {'hits':>5s} {'lift':>6s} {'recall':>7s} {'lift_back':>9s}")
for k, s in rows.items():
    if s is None:
        print(f"{k:30s}  (no threshold with >=30 alerts in-sample)"); continue
    print(f"{k:30s} {s['n']:6d} {s['hits']:5d} {s['lift']:6.2f} "
          f"{s['recall']:7.1%} {s['lift_back']:9.2f}")

p_veto = float((null_veto >= obs["lift"]).mean())
p_shift = float((null_shift >= obs["lift"]).mean())
mr = rows["vol only, matched rate"]["lift"]
amp_l = rows["AND gate (amp_only)"]["lift"] if rows["AND gate (amp_only)"] else np.nan

print("\nAUDIT VERDICTS")
print(f"  [1] beats matched-rate vol filter : {obs['lift']:.2f} vs {mr:.2f}  "
      f"-> {'PASS' if obs['lift'] > mr else 'FAIL'}")
print(f"  [2] beats random veto             : p = {p_veto:.3f}  "
      f"(null mean {null_veto.mean():.2f}) -> {'PASS' if p_veto < 0.05 else 'FAIL'}")
print(f"  [3] timing matters (shift null)   : p = {p_shift:.3f}  "
      f"(null mean {np.mean(null_shift):.2f}) -> {'PASS' if p_shift < 0.05 else 'FAIL'}")
print(f"  [4] beats amplitude-only control  : {obs['lift']:.2f} vs {amp_l:.2f}  "
      f"-> {'PASS' if obs['lift'] > amp_l else 'FAIL'}")
for nm in ["fano_randsign", "rand_antisym"]:
    s = rows[f"AND gate ({nm})"]
    if s:
        print(f"  [5] {nm:14s} retains {s['lift'] / obs['lift']:.0%} of fano lift")
print(f"  [6] reactive check: lift_back {obs['lift_back']:.2f} vs forward {obs['lift']:.2f} "
      f"-> {'mostly REACTIVE' if obs['lift_back'] > obs['lift'] else 'anticipatory'}")
print(f"  [7] recall: catches {obs['recall']:.0%} of holdout events")
if obs["hits"] < 10:
    print(f"\n  WARNING: only {obs['hits']} hits on holdout - lift is noisy at this count.")

json.dump({"in_sample": s_in, "holdout": rows, "p_veto": p_veto, "p_shift": p_shift,
           "randsign_signs": signs.tolist()},
          open(os.path.join(OUT, "audit_results.json"), "w"), indent=2, default=float)
print(f"\nSaved {OUT}/audit_results.json")
