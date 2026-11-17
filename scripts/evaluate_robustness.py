import os, time, json, sys
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score, precision_score, recall_score
from model_core import DualStreamV5, SEQ_LEN

torch.set_num_threads(8)
sys.stdout.reconfigure(line_buffering=True)

def f_beta(p, r, beta=0.5):
    d = (beta**2) * p + r
    return (1 + beta**2) * p * r / d if d > 0 else 0.0

def rolling_min(s, K):
    if K <= 1:
        return s
    pad = np.concatenate([np.full(K - 1, s[0], dtype=s.dtype), s])
    return np.lib.stride_tricks.sliding_window_view(pad, K).min(axis=1)

print("Loading checkpoint...", flush=True)
ckpt = torch.load("epoch=17-step=2178.ckpt", map_location="cpu")
model = DualStreamV5()
state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
filtered = {k: v for k, v in state_dict.items() if k in model.state_dict() and v.shape == model.state_dict()[k].shape}
model.load_state_dict(filtered, strict=False)
model.eval()

print("Loading Farm C dataset...", flush=True)
d = np.load("v4_pred_Wind_Farm_C.npy", allow_pickle=True).item()
feats_raw = d["feats"]  # (158384, 7)
status_raw = d["status"]
ds_raw = d["ds"]
ts_raw = pd.to_datetime(d["ts"])

# Build baseline 12-feature tensor (zeropad8)
feats12_base = np.zeros((len(feats_raw), 12), dtype=np.float32)
feats12_base[:, :7] = feats_raw
feats12_base[:, 7] = np.abs(feats_raw[:, 6] - feats_raw[:, 5]) * 0.2

# Pre-split into sequences
sequences = []
unique_ds = np.unique(ds_raw)
for j in unique_ds:
    m = (ds_raw == j)
    f_seq = feats12_base[m]
    st_seq = status_raw[m]
    ts_seq = ts_raw[m]
    if len(f_seq) >= SEQ_LEN:
        sequences.append({
            'ds': int(j),
            'feats': f_seq,
            'status': st_seq,
            'ts': ts_seq,
            'gt': (~np.isin(st_seq, [0, 2])).astype(int)[SEQ_LEN - 1:],
            'is_anom': bool(np.any(~np.isin(st_seq, [0, 2])))
        })

print(f"Total valid sequences: {len(sequences)}", flush=True)

feat_std = np.array([0.007, 0.0097, 0.0131, 0.0214, 0.001, 0.0009, 0.001, 0.0036], dtype=np.float32)
weights = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 3.0])
thr = 2.3263

def score_sequences(seq_list, pert_fn=None, K=24):
    seq_scores = []
    all_scores = []
    all_gts = []
    
    for s in seq_list:
        F = s['feats'].copy()
        if pert_fn is not None:
            F = pert_fn(F)
        
        w = np.lib.stride_tricks.sliding_window_view(F, window_shape=SEQ_LEN, axis=0)
        w = np.ascontiguousarray(w.transpose(0, 2, 1), dtype=np.float32)
        xt = torch.from_numpy(w).float()
        
        errs_list = []
        with torch.no_grad():
            for b in range(0, len(xt), 2048):
                xb = xt[b:b+2048]
                xr, _ = model(xb)
                diff = (xr[:, :, :8] - xb[:, :, :8]) ** 2
                w_diff = diff * weights
                errs_list.append(w_diff[:, -6:, :].mean(dim=1).numpy())
                
        err8 = np.concatenate(errs_list, axis=0)
        norm_err = np.clip(err8 / feat_std, 0, 20).mean(axis=1)
        score = rolling_min(norm_err, K)
        
        seq_scores.append({
            'score': score,
            'gt': s['gt'],
            'is_anom': s['is_anom'],
            'ts': s['ts'][SEQ_LEN - 1:]
        })
        all_scores.append(score)
        all_gts.append(s['gt'])
        
    return seq_scores, np.concatenate(all_scores), np.concatenate(all_gts)

def compute_metrics(seq_scores, all_sc, all_gt, threshold=thr):
    pr = (all_sc > threshold).astype(int)
    auc = float(roc_auc_score(all_gt, all_sc)) if len(np.unique(all_gt)) > 1 else 1.0
    prec = float(precision_score(all_gt, pr, zero_division=0))
    rec = float(recall_score(all_gt, pr, zero_division=0))
    norm_acc = float((pr[all_gt == 0] == 0).mean()) if (all_gt == 0).sum() > 0 else 1.0
    pointwise_fa = float((all_sc[all_gt == 0] > threshold).mean()) if (all_gt == 0).sum() > 0 else 0.0
    
    fb_list, ws_list, acc_list, evp, evt = [], [], [], [], []
    for ss in seq_scores:
        spr = (ss['score'] > threshold).astype(int)
        sgt = ss['gt']
        is_a = ss['is_anom']
        evp.append(int(spr.sum() > 0))
        evt.append(int(is_a))
        if is_a:
            tp = int(((spr == 1) & (sgt == 1)).sum())
            fp = int(((spr == 1) & (sgt == 0)).sum())
            fn = int(((spr == 0) & (sgt == 1)).sum())
            p_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            fb_list.append(f_beta(p_val, r_val))
            ws_list.append(float(r_val))
        else:
            acc_list.append(float((spr == 0).mean()))
            
    fbm = float(np.mean(fb_list)) if fb_list else 0.0
    wsm = float(np.mean(ws_list)) if ws_list else 0.0
    acm = float(np.mean(acc_list)) if acc_list else norm_acc
    
    evp = np.array(evp); evt = np.array(evt)
    tp_ev = int(((evp == 1) & (evt == 1)).sum())
    fp_ev = int(((evp == 1) & (evt == 0)).sum())
    fn_ev = int(((evp == 0) & (evt == 1)).sum())
    p_ev = tp_ev / (tp_ev + fp_ev) if (tp_ev + fp_ev) > 0 else 0.0
    r_ev = tp_ev / (tp_ev + fn_ev) if (tp_ev + fn_ev) > 0 else 0.0
    ef = f_beta(p_ev, r_ev)
    
    care = (fbm + wsm + ef + 2 * acm) / 5.0
    return {
        'care': round(care, 4),
        'auc': round(auc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'normal_acc': round(norm_acc, 4),
        'pointwise_fa_pct': round(pointwise_fa * 100.0, 3),
        'event_recall': round(r_ev, 4),
        'event_tp': tp_ev,
        'event_fn': fn_ev
    }

t0 = time.time()
print("Computing baseline evaluation...", flush=True)
base_seq_scores, base_sc, base_gt = score_sequences(sequences)
base_res = compute_metrics(base_seq_scores, base_sc, base_gt)
print(f"Baseline (elapsed {time.time()-t0:.1f}s): {base_res}", flush=True)

# ----------------- 1. Seasonal Generalization -----------------
print("\n--- Evaluating Seasonal Generalization ---", flush=True)
season_map = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11]
}

seasonal_results = {}
for sname, months in season_map.items():
    # Filter points matching season months
    s_sc_list, s_gt_list, s_seq_sub = [], [], []
    for ss in base_seq_scores:
        m = np.isin(ss['ts'].month, months)
        if m.sum() > 0:
            s_sc_list.append(ss['score'][m])
            s_gt_list.append(ss['gt'][m])
            s_seq_sub.append({
                'score': ss['score'][m],
                'gt': ss['gt'][m],
                'is_anom': bool((ss['gt'][m] == 1).any()),
                'ts': ss['ts'][m]
            })
    if s_sc_list:
        sub_sc = np.concatenate(s_sc_list)
        sub_gt = np.concatenate(s_gt_list)
        m_dict = compute_metrics(s_seq_sub, sub_sc, sub_gt)
        seasonal_results[sname] = {
            'steps': int(len(sub_sc)),
            'normal_steps': int((sub_gt == 0).sum()),
            'anomaly_steps': int((sub_gt == 1).sum()),
            'metrics': m_dict
        }
        print(f"Season {sname} ({len(sub_sc)} steps): CARE={m_dict['care']}, AUC={m_dict['auc']}, NormalAcc={m_dict['normal_acc']}, FA%={m_dict['pointwise_fa_pct']}%", flush=True)

# ----------------- 2. Missing Data Robustness -----------------
print("\n--- Evaluating Missing Data Robustness ---", flush=True)
missing_results = {}
for rate in [0.05, 0.10, 0.20]:
    t1 = time.time()
    def make_drop_fn(r):
        def drop_fn(F):
            rng = np.random.RandomState(42)
            mask = rng.rand(*F[:, :7].shape) < r
            F_pert = F.copy()
            for c in range(7):
                vals = F_pert[:, c]
                drop_idx = np.where(mask[:, c])[0]
                for idx in drop_idx:
                    vals[idx] = vals[idx - 1] if idx > 0 else 0.0
            return F_pert
        return drop_fn
    seq_sc, all_sc, all_gt = score_sequences(sequences, pert_fn=make_drop_fn(rate))
    m_dict = compute_metrics(seq_sc, all_sc, all_gt)
    missing_results[f"missing_{int(rate*100)}pct"] = m_dict
    print(f"Missing {int(rate*100)}% (elapsed {time.time()-t1:.1f}s): CARE={m_dict['care']}, AUC={m_dict['auc']}, Rec={m_dict['recall']}, FA%={m_dict['pointwise_fa_pct']}%", flush=True)

# ----------------- 3. Outlier Robustness -----------------
print("\n--- Evaluating Outlier Robustness ---", flush=True)
outlier_results = {}
for p_outlier in [0.01, 0.05, 0.10]:
    t1 = time.time()
    def make_outlier_fn(p):
        def outlier_fn(F):
            F_pert = F.copy()
            rng = np.random.RandomState(123)
            for c in [0, 2, 5, 6]:
                mask = rng.rand(len(F_pert)) < p
                F_pert[mask, c] += rng.uniform(0.5, 1.5, size=mask.sum())
            return F_pert
        return outlier_fn
    seq_sc, all_sc, all_gt = score_sequences(sequences, pert_fn=make_outlier_fn(p_outlier))
    m_dict = compute_metrics(seq_sc, all_sc, all_gt)
    outlier_results[f"outlier_{int(p_outlier*100)}pct"] = m_dict
    print(f"Outlier {int(p_outlier*100)}% (elapsed {time.time()-t1:.1f}s): CARE={m_dict['care']}, AUC={m_dict['auc']}, Rec={m_dict['recall']}, FA%={m_dict['pointwise_fa_pct']}%", flush=True)

all_experiments = {
    'baseline': base_res,
    'seasonal_generalization': seasonal_results,
    'missing_data_robustness': missing_results,
    'outlier_robustness': outlier_results
}

with open("robustness_and_seasonal_results.json", "w") as f:
    json.dump(all_experiments, f, indent=2)

print("\nSUCCESS: All missing experiments completed and saved to robustness_and_seasonal_results.json!", flush=True)
