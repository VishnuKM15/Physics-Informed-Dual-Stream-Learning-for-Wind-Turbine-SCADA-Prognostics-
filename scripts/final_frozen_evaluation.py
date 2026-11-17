import os, glob, time, json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score

torch.set_float32_matmul_precision('high')

SEQ_LEN = 36
BETA = 0.5
BASE_DIR = "data/raw/CARE_To_Compare"
FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Source-only selection space (deliberately small to avoid overfitting selection)
TEMPORAL_CANDIDATES = [('none',0), ('rmin',6), ('rmin',12), ('rmin',24), ('rmin',48)]
THRESH_PCTS = [95, 97, 99, 99.5]
INPUT_MODES = ['full12', 'zeropad8']   # hydraulic-ablation axis

def windows(feats, stride=1):
    T = len(feats)
    if T < SEQ_LEN:
        return np.empty((0, SEQ_LEN, feats.shape[1]), dtype=np.float32)
    w = np.lib.stride_tricks.sliding_window_view(feats, window_shape=SEQ_LEN, axis=0)
    return np.ascontiguousarray(w[::stride].transpose(0, 2, 1), dtype=np.float32)

def f_beta(p, r, beta=BETA):
    d = (beta**2) * p + r
    return (1 + beta**2) * p * r / d if d > 0 else 0.0

def rolling_min(s, K):
    if K <= 1:
        return s
    pad = np.concatenate([np.full(K - 1, s[0], dtype=s.dtype), s])
    return np.lib.stride_tricks.sliding_window_view(pad, K).min(axis=1)

def temporal_filter(s, spec):
    kind, k = spec
    if kind == 'none':
        return s
    if kind == 'rmin':
        return rolling_min(s, k)
    raise ValueError(spec)

def load_model():
    import importlib.util
    spec = importlib.util.spec_from_file_location("train_v5", "train_v5.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ckpts = sorted(glob.glob("ckpt5/*.ckpt"), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError("No checkpoint found in ckpt5/")
    ckpt = ckpts[-1]
    print(f"Loading checkpoint: {ckpt}")
    return mod.DualStreamV5.load_from_checkpoint(ckpt, map_location=DEVICE).to(DEVICE).eval()

def pred_path(farm):
    p = f'data/processed/v5_pred_{farm.replace(" ", "_")}.npy'
    if not os.path.exists(p):
        p = f'v5_pred_{farm.replace(" ", "_")}.npy'
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    return p

def compute_core8_errors(model, feats, input_mode):
    w = windows(feats, 1)
    if len(w) == 0:
        return np.empty((0, 8), dtype=np.float32)
    xt = torch.from_numpy(w).to(DEVICE).float()
    if input_mode == 'zeropad8':
        xt = torch.cat([xt[:, :, :8], torch.zeros_like(xt[:, :, 8:])], dim=-1)
    out = []
    with torch.no_grad():
        for i in range(0, len(xt), 2048):
            xb = xt[i:i + 2048]
            xr, _ = model(xb)
            sq = (xr[:, :, :8] - xb[:, :, :8]) ** 2
            out.append(sq[:, -6:, :].mean(1).cpu().numpy())  # last-hour time aggregation
    return np.concatenate(out).astype(np.float32)

def build_datasets(model, farm, input_mode):
    d = np.load(pred_path(farm), allow_pickle=True).item()
    e = pd.read_csv(os.path.join(BASE_DIR, farm, 'event_info.csv'), sep=None, engine='python')
    e.columns = e.columns.str.strip()
    ac = 'asset' if 'asset' in e.columns else 'asset_id'
    e['_asset'] = e[ac].astype(str).str.strip()
    e['event_start'] = pd.to_datetime(e['event_start']).astype('int64')
    e['event_end'] = pd.to_datetime(e['event_end']).astype('int64')
    out = []
    for j in np.unique(d['ds']):
        m = d['ds'] == j
        feats, ts, status = d['feats'][m], d['ts'][m], d['status'][m]
        asset = str(d['asset'][m][0]).strip()
        if len(feats) < SEQ_LEN:
            continue
        errs = compute_core8_errors(model, feats, input_mode)
        ts_o = ts[SEQ_LEN - 1:]
        gt = (~np.isin(status, [0, 2])).astype(int)[SEQ_LEN - 1:]
        e2 = e[e['_asset'] == asset]
        t0n, t1n = ts_o.min(), ts_o.max()
        if len(e2):
            ov = (e2['event_start'].values <= t1n) & (e2['event_end'].values >= t0n)
            an = e2[ov]
            an = an[an['event_label'].astype(str).str.strip() == 'anomaly']
        else:
            an = e2
        is_anom = len(an) > 0
        out.append(dict(asset=asset, ts=ts_o, errs=errs, gt=gt, is_anom=is_anom,
                        ev_start=an['event_start'].values[0] if is_anom else None,
                        ev_end=an['event_end'].values[0] if is_anom else None))
    return out

def care_at(ds_scores, thr):
    fb, ws, acc, evp, evt = [], [], [], [], []
    for dd in ds_scores:
        pr = (dd['score'] > thr).astype(int); gt = dd['gt']
        evp.append(int(pr.sum() > 0)); evt.append(int(dd['is_anom']))
        if dd['is_anom']:
            tp = int(((pr == 1) & (gt == 1)).sum()); fp = int(((pr == 1) & (gt == 0)).sum())
            fn = int(((pr == 0) & (gt == 1)).sum())
            fb.append(f_beta(tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0))
            if dd.get('ev_start') is not None:
                mk = (dd['ts'] >= dd['ev_start']) & (dd['ts'] <= dd['ev_end'])
                if mk.sum():
                    pos = (dd['ts'][mk] - dd['ev_start']) / max(dd['ev_end'] - dd['ev_start'], 1)
                    wt = np.where(pos <= 0.5, 1, np.clip(1 - (pos - 0.5)/0.5, 0, 1))
                    ws.append(float((wt * pr[mk]).sum() / max(wt.sum(), 1)))
        else:
            acc.append(float((pr == gt).mean()))
    fbm = float(np.mean(fb)) if fb else 0.0
    wsm = float(np.mean(ws)) if ws else 0.0
    acm = float(np.mean(acc)) if acc else 0.0
    evp = np.array(evp); evt = np.array(evt)
    tp = int(((evp == 1) & (evt == 1)).sum()); fp = int(((evp == 1) & (evt == 0)).sum())
    fn = int(((evp == 0) & (evt == 1)).sum())
    ef = f_beta(tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0)
    care = 0.0 if evp.sum() == 0 else (acm if acm < 0.5 else (fbm + wsm + ef + 2*acm)/5)
    return dict(care=care, fb=fbm, ws=wsm, ef=ef, acc=acm, tp=tp, fp=fp, fn=fn)

def apply_pipeline(datasets, med, mad, temp_spec):
    res = []
    for dd in datasets:
        z = np.clip((dd['errs'] - med) / (mad + 1e-9), 0, 20).astype(np.float32)
        s = z.mean(1)                      # feature aggregation = mean over core-8
        s = temporal_filter(s, temp_spec)  # causal persistence filter (within dataset)
        res.append({**dd, 'score': s})
    return res

def main():
    t0 = time.time()
    print("=" * 95)
    print("FINAL FROZEN ZERO-SHOT EVALUATION")
    print("P1 source-only selection | P2 hydraulic ablation | P3 grouping check | P4 single frozen eval")
    print("=" * 95)
    print(f"DEVICE: {DEVICE}")
    model = load_model()

    # ---------------- PRIORITY 3 ----------------
    print("\n" + "-" * 95)
    print("PRIORITY 3: CHRONOLOGICAL GROUPING VERIFICATION")
    print("-" * 95)
    violations = 0; total_ds = 0
    for farm in FARMS:
        d = np.load(pred_path(farm), allow_pickle=True).item()
        for j in np.unique(d['ds']):
            ts = d['ts'][d['ds'] == j]
            total_ds += 1
            if len(ts) > 1 and not np.all(np.diff(ts) > 0):
                violations += 1
                print(f"  VIOLATION: {farm} ds={j} not strictly increasing")
    print(f"  Checked {total_ds} turbine sequences -> chronological violations: {violations}")
    print(f"  Windows & rolling alarms are computed per dataset -> cannot cross boundaries.")

    results_by_mode = {}
    for input_mode in INPUT_MODES:
        print("\n" + "=" * 95)
        print(f"HYDRAULIC ABLATION ARM: input_mode = {input_mode}")
        print("=" * 95)
        src = build_datasets(model, FARMS[0], input_mode) + build_datasets(model, FARMS[1], input_mode)
        tgt = build_datasets(model, FARMS[2], input_mode)
        print(f"  Source(A+B) datasets={len(src)} (anomalous={sum(x['is_anom'] for x in src)})  "
              f"Target(C) datasets={len(tgt)} (anomalous={sum(x['is_anom'] for x in tgt)})")

        # normalizer fitted on SOURCE normal only
        src_norm = np.concatenate([x['errs'][x['gt'] == 0] for x in src])
        med = np.median(src_norm, axis=0)
        mad = np.median(np.abs(src_norm - med), axis=0) * 1.4826
        print(f"  Robust normalizer fitted on {len(src_norm)} source-normal windows.")

        # ---------------- PRIORITY 1: source-only selection ----------------
        print("  PRIORITY 1: selecting config + threshold on SOURCE (A+B) only ...")
        best = None
        for temp_spec in TEMPORAL_CANDIDATES:
            src_scored = apply_pipeline(src, med, mad, temp_spec)
            src_normal_scores = np.concatenate([x['score'][x['gt'] == 0] for x in src_scored])
            for pct in THRESH_PCTS:
                thr = float(np.percentile(src_normal_scores, pct))
                r = care_at(src_scored, thr)
                if best is None or r['care'] > best['sel']:
                    best = dict(sel=r['care'], temp=temp_spec, pct=pct, thr=thr, src_care=r['care'])
        print(f"    SELECTED (source-only): temporal={best['temp']}  thr_pct={best['pct']}  "
              f"thr={best['thr']:.4f}  source_CARE={best['src_care']:.4f}")

        # ---------------- PRIORITY 4: frozen single eval on C ----------------
        tgt_scored = apply_pipeline(tgt, med, mad, best['temp'])
        r = care_at(tgt_scored, best['thr'])
        allsc = np.concatenate([x['score'] for x in tgt_scored])
        allgt = np.concatenate([x['gt'] for x in tgt_scored])
        prc = (allsc > best['thr']).astype(int)
        auc = roc_auc_score(allgt, allsc)
        prec = precision_score(allgt, prc, zero_division=0)
        rec = recall_score(allgt, prc, zero_division=0)
        norm_ds = [x for x in tgt_scored if not x['is_anom']]
        fa_ds = sum(1 for x in norm_ds if (x['score'] > best['thr']).any())
        n_norm_pts = int((allgt == 0).sum())
        n_fp_pts = int(((allsc > best['thr']) & (allgt == 0)).sum())

        print(f"\n  FROZEN FARM C RESULT ({input_mode}):")
        print(f"    CARE                  = {r['care']:.4f}")
        print(f"    F-beta (pointwise)    = {r['fb']:.4f}")
        print(f"    WS (earliness)        = {r['ws']:.4f}")
        print(f"    EF-beta (event)       = {r['ef']:.4f}")
        print(f"    Acc (normal)          = {r['acc']:.4f}")
        print(f"    AUC                   = {auc:.4f}")
        print(f"    Precision (pointwise) = {prec:.4f}")
        print(f"    Recall (pointwise)    = {rec:.4f}")
        print(f"    Event TP/FP/FN        = {r['tp']}/{r['fp']}/{r['fn']}")
        print(f"    Normal datasets w/ >=1 false alarm = {fa_ds}/{len(norm_ds)}")
        print(f"    Pointwise false alarms = {n_fp_pts}/{n_norm_pts} ({100*n_fp_pts/max(n_norm_pts,1):.3f}%)")

        results_by_mode[input_mode] = dict(
            care=r['care'], auc=auc, prec=prec, rec=rec, ef=r['ef'], fb=r['fb'],
            ws=r['ws'], acc=r['acc'], tp=r['tp'], fp=r['fp'], fn=r['fn'],
            fa_datasets=fa_ds, n_norm_ds=len(norm_ds),
            selected_temp=str(best['temp']), selected_pct=best['pct'],
            selected_thr=best['thr'], src_care=best['src_care'])

    # ---------------- PRIORITY 2 summary ----------------
    print("\n" + "=" * 95)
    print("PRIORITY 2 SUMMARY: HYDRAULIC ABLATION (identical source-only protocol)")
    print("=" * 95)
    print(f"{'input':<11}{'CARE':>7}{'AUC':>7}{'Prec':>7}{'Rec':>7}{'EF':>7}{'Acc':>7}"
          f"{'TP':>4}{'FP':>4}{'FN':>4}{'FA-ds':>8}{'sel_temp':>14}{'sel_pct':>9}")
    for im in INPUT_MODES:
        m = results_by_mode[im]
        print(f"{im:<11}{m['care']:>7.4f}{m['auc']:>7.4f}{m['prec']:>7.4f}{m['rec']:>7.4f}"
              f"{m['ef']:>7.4f}{m['acc']:>7.4f}{m['tp']:>4}{m['fp']:>4}{m['fn']:>4}"
              f"{str(m['fa_datasets'])+'/'+str(m['n_norm_ds']):>8}{m['selected_temp']:>14}{m['selected_pct']:>9}")

    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/final_frozen_eval.json', 'w') as f:
        json.dump(results_by_mode, f, indent=2)
    print("\nSaved: results/metrics/final_frozen_eval.json")
    print(f"Elapsed: {time.time()-t0:.0f}s")
    print("Model selection used ONLY Farm A+B. Farm C was evaluated exactly once, frozen.")

if __name__ == '__main__':
    main()