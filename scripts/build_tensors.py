#!/usr/bin/env python3
"""
CARE v5 -- Physics-Subsystem-Aligned Tensor Builder
===================================================
Self-contained: reads ONLY raw dataset + feature_description.csv.
No dependency on generated category CSVs.

Outputs 12 features in two physical streams:
  Aero/power boundary : [wind, rotor, power, pitch, ambient, power_residual]
  Thermal/aux state   : [oil_rise, brg_max_rise, brg_spread,
                         electrical, hydraulic, hydraulic_mask]

The semantically broken generator proxy is intentionally EXCLUDED.
"""
from __future__ import annotations

import csv
import json
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw" / "CARE_To_Compare"
PROC = ROOT / "data" / "processed"
CONF = ROOT / "configs"
METRICS = ROOT / "results" / "metrics"

FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]
SEQ_LEN = 36
STRIDE = 9
META = ["time_stamp", "asset_id", "train_test", "status_type_id"]
N_FEATURES = 12

STAT_SUFFIX = {
    "average": "avg", "maximum": "max",
    "minimum": "min", "std_dev": "std",
}

# ---------------------------------------------------------------- rules
# Description-based concept rules. These are applied to the RAW
# feature_description.csv descriptions, not to guessed sensor indices.
def _r_wind(d):
    return ("wind" in d and ("speed" in d) and
            "estimated" not in d and "direction" not in d)

def _r_rotor(d):
    return ("rotor" in d and ("rpm" in d or "speed" in d) and
            "pitch" not in d and "yaw" not in d and "power" not in d)

def _r_power(d):
    bad = ["reactive", "total", "energy", "kwh", "aeration", "internal",
           "setpoint", "limitation", "possible", "apparent", "kvar"]
    return (("active power" in d or "grid power" in d) and
            not any(b in d for b in bad))

def _r_pitch(d):
    bad = ["current", "motor", "aeration", "speed", "power", "pressure"]
    return ("pitch" in d and "angle" in d and not any(b in d for b in bad))

def _r_ambient(d):
    return "ambient" in d and "temp" in d

def _r_oil(d):
    bad = ["level", "pressure", "pump", "delta", "differential", "heater"]
    return ("oil" in d and "temp" in d and not any(b in d for b in bad))

def _r_bearing(d):
    return ("bearing" in d and "temp" in d and "generator" not in d and
            any(b in d for b in ["gearbox", "planetary", "gear", "main"]))

def _r_current(d):
    bad = ["rotor", "pitch", "aeration", "motor", "reactive"]
    return ("current" in d and "phase" in d and not any(b in d for b in bad))

def _r_hydraulic(d):
    bad = ["temp", "temperature", "level", "pump", "heater"]
    return (("hydraulic" in d or "pressure" in d) and
            not any(b in d for b in bad))

RULES = {
    "wind":      (_r_wind,      "single"),
    "rotor":     (_r_rotor,     "single"),
    "power":     (_r_power,     "single"),
    "pitch":     (_r_pitch,     "single"),
    "ambient":   (_r_ambient,   "single"),
    "oil":       (_r_oil,       "single"),
    "bearing":   (_r_bearing,   "many"),
    "current":   (_r_current,   "many"),
    "hydraulic": (_r_hydraulic, "many"),
}

# Physical plausibility: (median_ok, p99_ok)
def _p_temp(med, q99):
    return -10 <= med <= 95 and q99 <= 160

def _p_power(med, q99):
    return -0.05 <= med <= 1.2 and q99 <= 1.5

def _p_wind(med, q99):
    return 2 <= med <= 16 and q99 <= 40

def _p_rotor(med, q99):
    return 2 <= med <= 16 and q99 <= 22

def _p_current(med, q99):
    return 0 <= med <= 2000 and q99 <= 3000

def _p_hyd(med, q99):
    return -50 <= med <= 600 and q99 <= 1500

PLAUS = {
    "wind": _p_wind, "rotor": _p_rotor, "power": _p_power,
    "pitch": lambda m, q: -10 <= m <= 40 and q <= 100,
    "ambient": _p_temp, "oil": _p_temp, "bearing": _p_temp,
    "current": _p_current, "hydraulic": _p_hyd,
}

# Optional manual overrides: configs/feature_overrides.json
# Format: {"Wind Farm A": {"oil": ["sensor_12_avg"], ...}}
OVERRIDE_PATH = CONF / "feature_overrides.json"
OVERRIDES = json.loads(OVERRIDE_PATH.read_text()) if OVERRIDE_PATH.exists() else {}


# ---------------------------------------------------------------- IO
def sniff_delim(path: Path) -> str:
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        sample = f.read(8192)
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except Exception:
        first = sample.splitlines()[0] if sample.splitlines() else ""
        counts = {d: first.count(d) for d in [",", ";", "\t", "|"]}
        best = max(counts, key=counts.get)
        return best if counts[best] > 0 else ","


def read_table(path: Path, usecols=None, nrows=None) -> pd.DataFrame:
    """Robust reader: C engine first, Python engine as fallback."""
    delim = sniff_delim(path)
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "latin-1"]:
        try:
            df = pd.read_csv(path, sep=delim, encoding=enc, engine="c",
                             on_bad_lines="skip", usecols=usecols,
                             nrows=nrows, low_memory=False)
            df.columns = df.columns.astype(str).str.strip()
            if df.shape[1] > 0:
                return df
        except Exception as e:
            last_err = e
    # Slow but tolerant fallback
    df = pd.read_csv(path, sep=delim, engine="python",
                     encoding="utf-8", encoding_errors="replace",
                     on_bad_lines="skip", usecols=usecols, nrows=nrows)
    df.columns = df.columns.astype(str).str.strip()
    return df


def build_lookup(farm: str) -> dict:
    """Build column -> metadata directly from the ground-truth description file."""
    path = RAW / farm / "feature_description.csv"
    df = read_table(path)
    needed = {"sensor_name", "statistics_type", "description", "unit"}
    missing = needed - set(df.columns)
    if missing:
        raise RuntimeError(f"{path}: missing columns {missing}; got {list(df.columns)}")

    lookup = {}
    for _, row in df.iterrows():
        name = str(row["sensor_name"]).strip()
        stats = str(row["statistics_type"]).split(",")
        for stat in stats:
            suffix = STAT_SUFFIX.get(stat.strip())
            if suffix is None:
                continue
            col = f"{name}_{suffix}"
            lookup[col] = {
                "description": str(row["description"]),
                "unit": str(row["unit"]),
            }
    return lookup


def normalize_asset(x) -> str:
    """'11.0' and '11' and ' 11 ' must all become '11' for event matching."""
    s = str(x).strip()
    try:
        f = float(s)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except (ValueError, TypeError):
        pass
    return s


# ---------------------------------------------------------------- mapping
def resolve_concepts(farm: str, lookup: dict, sample: pd.DataFrame) -> dict:
    header = set(sample.columns)
    resolved = {}

    for concept, (rule, mode) in RULES.items():
        # 1. Explicit override takes precedence
        ov = OVERRIDES.get(farm, {}).get(concept)
        if ov:
            cols = [c for c in ov if c in header]
            if cols:
                resolved[concept] = {
                    "columns": cols,
                    "description": lookup.get(cols[0], {}).get("description", "OVERRIDE"),
                    "source": "override",
                }
                continue

        # 2. Description match
        candidates = []
        for col, meta in lookup.items():
            if not col.endswith("_avg") or col not in header:
                continue
            desc = meta["description"].lower()
            if not rule(desc):
                continue

            vals = pd.to_numeric(sample[col], errors="coerce").replace(
                [np.inf, -np.inf], np.nan).dropna().values
            if len(vals) < 30:
                continue
            med = float(np.median(vals))
            q99 = float(np.quantile(vals, 0.99))
            if not PLAUS[concept](med, q99):
                continue

            # Prefer explicitly named sensors over generic sensor_N
            named = not col.startswith("sensor_")
            candidates.append({
                "column": col, "description": meta["description"],
                "unit": meta["unit"], "median": med, "p99": q99,
                "named": named,
            })

        if not candidates:
            resolved[concept] = {"columns": [], "description": "", "source": "none"}
            continue

        # Deterministic ranking
        candidates.sort(key=lambda c: (
            not c["named"],
            0 if ("gearbox" in c["description"].lower() and concept in ("oil", "bearing")) else 1,
            c["column"],
        ))

        if mode == "single":
            chosen = candidates[:1]
        else:
            chosen = candidates

        resolved[concept] = {
            "columns": [c["column"] for c in chosen],
            "description": chosen[0]["description"],
            "unit": chosen[0]["unit"],
            "median": chosen[0]["median"],
            "p99": chosen[0]["p99"],
            "n_candidates": len(candidates),
            "source": "description+plausibility",
        }
    return resolved


# ---------------------------------------------------------------- features
def _num(df, col):
    if col is None or col not in df.columns:
        return np.full(len(df), np.nan, dtype=np.float64)
    return pd.to_numeric(df[col], errors="coerce").values.astype(np.float64)


def compute_features(df: pd.DataFrame, resolved: dict) -> np.ndarray:
    n = len(df)

    def get(concept, idx=0):
        cols = resolved.get(concept, {}).get("columns", [])
        return _num(df, cols[idx] if len(cols) > idx else None)

    wind = np.clip(np.nan_to_num(get("wind"), nan=0.0), 0, 35)
    rotor = np.clip(np.nan_to_num(get("rotor"), nan=0.0), 0, 20)
    power = np.clip(np.nan_to_num(get("power"), nan=0.0), -0.1, 1.1)
    pitch = np.clip(np.nan_to_num(get("pitch"), nan=0.0), -5, 95)

    amb_raw = get("ambient")
    amb_fill = np.nanmedian(amb_raw) if np.isfinite(amb_raw).any() else 15.0
    amb = np.clip(np.nan_to_num(amb_raw, nan=amb_fill), -50, 60)

    oil_raw = get("oil")
    oil = np.where(np.isfinite(oil_raw), np.clip(oil_raw, -50, 150), amb)

    brg_cols = resolved.get("bearing", {}).get("columns", [])
    if brg_cols:
        mat = np.column_stack([_num(df, c) for c in brg_cols])
        mat = np.where(np.isfinite(mat), np.clip(mat, -50, 150), amb[:, None])
        brg_max = mat.max(axis=1)
        brg_min = mat.min(axis=1)
    else:
        brg_max = amb.copy()
        brg_min = amb.copy()

    cur_cols = resolved.get("current", {}).get("columns", [])
    if cur_cols:
        cur = np.column_stack([_num(df, c) for c in cur_cols])
        cur = np.nan_to_num(np.clip(cur, 0, 2500), nan=0.0)
        electrical = cur.max(axis=1) / 2500.0
    else:
        electrical = np.zeros(n)

    hyd_cols = resolved.get("hydraulic", {}).get("columns", [])
    if hyd_cols:
        hyd = np.column_stack([_num(df, c) for c in hyd_cols])
        hyd = np.nan_to_num(np.clip(hyd, 0, 1500), nan=0.0)
        hydraulic = np.clip(hyd.mean(axis=1) / 500.0, 0, 1)
        hyd_mask = np.ones(n)
    else:
        hydraulic = np.zeros(n)
        hyd_mask = np.zeros(n)

    # Physics residual: power curve saturation shape (v/12)^3, capped at 1
    ideal = np.minimum(1.0, (wind / 12.0) ** 3)
    power_residual = np.clip(power - ideal, -0.5, 0.5)

    feats = np.stack([
        wind / 25.0,
        rotor / 20.0,
        power,                                   # ALREADY per-unit: do NOT divide
        pitch / 90.0,
        amb / 100.0,
        np.clip(oil - amb, -20, 100) / 100.0,
        np.clip(brg_max - amb, -20, 120) / 100.0,
        np.clip(brg_max - brg_min, 0, 80) / 50.0,
        power_residual,
        electrical,
        hydraulic,
        hyd_mask,
    ], axis=1).astype(np.float32)

    return np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)


# ---------------------------------------------------------------- parsing
def parse_turbine(task):
    path_str, farm, resolved = task
    path = Path(path_str)
    try:
        header = read_table(path, nrows=0).columns.tolist()
        needed = list(META)
        for spec in resolved.values():
            needed.extend(spec.get("columns", []))
        needed = list(dict.fromkeys(c for c in needed if c in header))

        df = read_table(path, usecols=needed)
        for m in META:
            if m not in df.columns:
                raise KeyError(f"missing metadata column '{m}'")

        df["time_stamp"] = pd.to_datetime(df["time_stamp"], errors="coerce")
        df = df.dropna(subset=["time_stamp"]).sort_values("time_stamp").reset_index(drop=True)
        if len(df) < SEQ_LEN:
            return None

        status = pd.to_numeric(df["status_type_id"], errors="coerce").fillna(0).astype(np.int32).values
        tt = df["train_test"].astype(str).str.strip().str.lower().values
        asset = normalize_asset(df["asset_id"].iloc[0])

        feats = compute_features(df, resolved)

        return {
            "farm": farm, "asset": asset,
            "ts": df["time_stamp"].values.astype("int64"),
            "status": status, "tt": tt, "feats": feats,
        }
    except Exception:
        return {"error": f"{path}\n{traceback.format_exc()}"}


def make_windows(feats: np.ndarray) -> np.ndarray:
    if len(feats) < SEQ_LEN:
        return np.empty((0, SEQ_LEN, N_FEATURES), dtype=np.float32)
    w = np.lib.stride_tricks.sliding_window_view(feats, SEQ_LEN, axis=0)
    return np.ascontiguousarray(w[::STRIDE].transpose(0, 2, 1), dtype=np.float32)


# ---------------------------------------------------------------- main
def main():
    t0 = time.time()
    print("=" * 100)
    print("CARE v5 -- SELF-VALIDATING PHYSICS-SUBSYSTEM TENSOR BUILDER")
    print("=" * 100)
    if not RAW.is_dir():
        raise RuntimeError(f"Raw dataset not found at {RAW}")

    PROC.mkdir(parents=True, exist_ok=True)
    METRICS.mkdir(parents=True, exist_ok=True)
    CONF.mkdir(parents=True, exist_ok=True)

    # ---------- Phase 1: build mappings + SMOKE TEST (serial, full traceback)
    lookups, mappings, smoke = {}, {}, {}
    for farm in FARMS:
        print(f"\n--- {farm} ---")
        lookup = build_lookup(farm)
        lookups[farm] = lookup
        print(f"  Ground-truth lookup: {len(lookup)} columns")

        files = sorted((RAW / farm / "datasets").glob("*.csv"))
        if not files:
            raise RuntimeError(f"No CSVs in {RAW / farm / 'datasets'}")

        print(f"  Smoke-testing parser on {files[0].name} ...")
        sample = read_table(files[0], nrows=5000)
        print(f"  Parsed OK: {sample.shape}")

        resolved = resolve_concepts(farm, lookup, sample)
        mappings[farm] = resolved

        for concept, spec in resolved.items():
            cols = spec.get("columns", [])
            desc = spec.get("description", "")
            med = spec.get("median", float("nan"))
            q99 = spec.get("p99", float("nan"))
            print(f"    {concept:<10} n={len(cols):<3} "
                  f"{','.join(cols)[:52]:<54} med={med:9.2f} p99={q99:9.2f}")
            print(f"               {desc[:95]}")

        # Real end-to-end smoke test through the SAME code path as parallel workers
        test = parse_turbine((str(files[0]), farm, resolved))
        if test is None or "error" in test:
            print("\nSMOKE TEST FAILED -- full traceback:")
            print(test if test is None else test["error"])
            raise SystemExit(1)
        print(f"  End-to-end smoke test OK: {test['feats'].shape}, "
              f"asset={test['asset']}, finite={np.isfinite(test['feats']).mean():.4f}")
        smoke[farm] = test

    (CONF / "v5_feature_map.json").write_text(json.dumps(mappings, indent=2))

    # ---------- Phase 2: parallel parse of ALL turbines
    tasks = []
    for farm in FARMS:
        for f in sorted((RAW / farm / "datasets").glob("*.csv")):
            tasks.append((str(f), farm, mappings[farm]))

    print(f"\nParsing {len(tasks)} turbines ...")
    workers = min(4, os_cpu := (os_cpu_count := __import__("os").cpu_count()) or 2)
    with ProcessPoolExecutor(max_workers=workers) as ex:
        parsed = list(ex.map(parse_turbine, tasks))

    valid = [p for p in parsed if isinstance(p, dict) and "error" not in p and p is not None]
    errors = [p["error"] for p in parsed if isinstance(p, dict) and "error" in p]

    print(f"  Success: {len(valid)} / {len(tasks)}")
    if errors:
        print(f"  FIRST ERROR (full traceback):")
        print(errors[0])
        if len(valid) < 0.9 * len(tasks):
            raise SystemExit(1)

    # ---------- Phase 3: assemble
    trX, trF = [], []
    preds = {farm: {"ts": [], "status": [], "feats": [], "asset": [], "ds": []}
             for farm in FARMS}

    for p in valid:
        farm = p["farm"]
        fi = FARMS.index(farm)

        train_mask = (p["tt"] == "train") & np.isin(p["status"], [0, 2])
        if train_mask.sum() >= SEQ_LEN:
            w = make_windows(p["feats"][train_mask])
            if len(w):
                trX.append(w)
                trF.append(np.full(len(w), fi, dtype=np.int64))

        pred_mask = (p["tt"] == "prediction")
        if pred_mask.any():
            d = preds[farm]
            d["ts"].append(p["ts"][pred_mask])
            d["status"].append(p["status"][pred_mask])
            d["feats"].append(p["feats"][pred_mask])
            d["asset"].append(np.full(int(pred_mask.sum()), p["asset"], dtype=object))
            d["ds"].append(np.full(int(pred_mask.sum()), len(d["ts"]) - 1, dtype=np.int64))

    if not trX:
        raise RuntimeError("No training windows produced.")

    X = np.concatenate(trX)
    F = np.concatenate(trF)

    # ---------- Phase 4: VALIDATION GATE
    print("\n" + "=" * 100)
    print("VALIDATION GATE")
    print("=" * 100)
    names = ["wind", "rotor", "power_pu", "pitch", "ambient",
             "oil_rise", "brg_max_rise", "brg_spread",
             "power_residual", "electrical", "hydraulic", "hyd_mask"]

    sample = X[::max(1, len(X) // 20000)].reshape(-1, N_FEATURES)
    print(f"\n{'Feature':<18}{'Mean':>10}{'Std':>10}{'Min':>10}{'Max':>10}")
    print("-" * 60)
    for i, name in enumerate(names):
        v = sample[:, i]
        print(f"{name:<18}{v.mean():>10.4f}{v.std():>10.4f}{v.min():>10.4f}{v.max():>10.4f}")

    problems = []
    if not np.isfinite(X).all():
        problems.append("Non-finite values in training tensor")
    if sample[:, 2].std() < 0.05:
        problems.append(f"ACTIVE POWER DEGENERATE (std={sample[:, 2].std():.6f})")
    if sample[:, 5].std() < 0.005:
        problems.append("Oil rise degenerate")
    if sample[:, 6].std() < 0.005:
        problems.append("Bearing rise degenerate")

    print("\nPer-farm availability / health:")
    for fi, farm in enumerate(FARMS):
        sub = X[F == fi].reshape(-1, N_FEATURES)
        print(f"  {farm}: n={int((F == fi).sum())} "
              f"power_std={sub[:, 2].std():.4f} "
              f"brg_std={sub[:, 6].std():.4f} "
              f"hyd_mask={sub[:, 11].mean():.3f}")

    if problems:
        print("\nFAILED VALIDATION:")
        for p in problems:
            print("  -", p)
        raise SystemExit(1)

    # ---------- Phase 5: save
    np.save(PROC / "v5_train_X.npy", X)
    np.save(PROC / "v5_train_F.npy", F)

    for farm in FARMS:
        d = preds[farm]
        if not d["ts"]:
            continue
        out = {k: np.concatenate(v) for k, v in d.items()}
        np.save(PROC / f"v5_pred_{farm.replace(' ', '_')}.npy", out, allow_pickle=True)

    report = {
        "shape": list(X.shape),
        "farm_counts": {f: int((F == i).sum()) for i, f in enumerate(FARMS)},
        "feature_names": names,
        "streams": {
            "aero_boundary": [0, 1, 2, 3, 4, 8],
            "thermal_aux": [5, 6, 7, 9, 10, 11],
        },
        "design_notes": [
            "Active power is already per-unit in ALL farms; no /3000.",
            "Generator thermal proxy EXCLUDED (semantically inconsistent).",
            "Temperatures used as rise above ambient, hard-clipped.",
            "Bearing MAX and SPREAD replace bearing MEAN.",
            "Hydraulic availability mask handles absent sensors.",
            "Asset IDs normalized so event_info joins are exact.",
        ],
    }
    (METRICS / "v5_verification.json").write_text(json.dumps(report, indent=2))

    print(f"\nSaved {X.shape} training tensor and 3 prediction tensors.")
    print(f"Elapsed: {time.time() - t0:.1f}s")
    print("VALIDATION PASSED -- data foundation is sound.")


if __name__ == "__main__":
    main()