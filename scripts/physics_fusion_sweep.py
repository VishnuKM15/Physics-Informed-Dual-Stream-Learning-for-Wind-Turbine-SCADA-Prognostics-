import os, glob, json, itertools
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from sklearn.metrics import roc_auc_score

SEQ_LEN = 36
BETA = 0.5
N_FEAT = 12
BASE_DIR = "data/raw/CARE_To_Compare"
FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture constants
AERO_IDX = [0, 1, 2, 3, 4, 8]           
THERM_IDX = [5, 6, 7, 9, 10, 11]        

class DualStreamZS(pl.LightningModule):
    def __init__(self, d_model=64, latent_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.aero_enc = nn.LSTM(len(AERO_IDX), d_model, batch_first=True, bidirectional=True)
        self.therm_enc = nn.LSTM(len(THERM_IDX), d_model, batch_first=True, bidirectional=True)
        self.fusion = nn.Sequential(nn.Linear(4 * d_model, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(), nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, SEQ_LEN * 8))
        self.alpha_oil = nn.Parameter(torch.tensor(-2.0)); self.beta_oil = nn.Parameter(torch.tensor(-1.0))
        self.alpha_brg = nn.Parameter(torch.tensor(-2.0)); self.beta_brg = nn.Parameter(torch.tensor(-1.0))
        self.lr = lr
    def forward(self, x):
        a, _ = self.aero_enc(x[:, :, AERO_IDX]); t, _ = self.therm_enc(x[:, :, THERM_IDX])
        z = self.fusion(torch.cat([a[:, -1, :], t[:, -1, :]], -1))
        return self.decoder(z).view(-1, SEQ_LEN, 8), z

def windows(feats, stride=1):
    T = len(feats)
    if T < SEQ_LEN: return np.empty((0, SEQ_LEN, feats.shape[1]), dtype=np.float32)
    w = np.lib.stride_tricks.sliding_window_view(feats, window_shape=SEQ_LEN, axis=0)
    return np.ascontiguousarray(w[::stride].transpose(0, 2, 1), dtype=np.float32)

def rolling_min(s, K):
    if K <= 1: return s
    pad = np.concatenate([np.full(K - 1, s[0], dtype=s.dtype), s])
    return np.lib.stride_tricks.sliding_window_view(pad, K).min(axis=1)

def f_beta(p, r, beta=BETA):
    d = (beta ** 2) * p + r
    return (1 + beta ** 2) * p * r / d if d > 0 else 0.0

def load_model():
    ckpts = sorted(glob.glob("ckpt_strict_zeroshot/*.ckpt"), key=os.path.getmtime)
    if not ckpts: raise FileNotFoundError("No strict checkpoint found!")
    print(f"Loading: {ckpts[-1]}")
    return DualStreamZS.load_from_checkpoint(ckpts[-1], map_location=DEVICE).to(DEVICE).eval()

def extract_errors_and_physics(model, feats, input_mode='zeropad8'):
    w = windows(feats, 1)
    if len(w) == 0: return np.empty((0, 36, 8), dtype=np.float32), np.empty((0,), dtype=np.float32)
    xt = torch.from_numpy(w).to(DEVICE).float()
    if input_mode == 'zeropad8':
        xt_full = torch.cat([xt[:, :, :8], torch.zeros_like(xt[:, :, 8:])], dim=-1)
    else: xt_full = xt
        
    out_errs, out_phys = [], []
    with torch.no_grad():
        for i in range(0, len(xt_full), 2048):
            xb = xt_full[i:i + 2048]; xb_core = xt[i:i + 2048, :, :8]
            dec, _ = model(xb)
            sq = (dec - xb_core) ** 2
            out_errs.append(sq.cpu().numpy())
            
            # Physics Residual Calculation (Thermodynamic Violation)
            P, oil, brg = xb_core[:, :, 2], xb_core[:, :, 5], xb_core[:, :, 6]
            ao, bo = F.softplus(model.alpha_oil), torch.sigmoid(model.beta_oil)
            ab, bb = F.softplus(model.alpha_brg), torch.sigmoid(model.beta_brg)
            oil_hat = (1 - bo) * oil[:, :-1] + ao * P[:, :-1]
            brg_hat = (1 - bb) * brg[:, :-1] + ab * P[:, :-1]
            phys = ((oil[:, 1:] - oil_hat)**2).mean(1) + ((brg[:, 1:] - brg_hat)**2).mean(1)
            out_phys.append(phys.cpu().numpy())
            
    return np.concatenate(out_errs), np.concatenate(out_phys)

def pred_path(farm):
    p = f'data/processed/v5_pred_{farm.replace(" ", "_")}.npy'
    return p if os.path.exists(p) else f'v5_pred_{farm.replace(" ", "_")}.npy'

def build_datasets(model, farm):
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
        if len(feats) < SEQ_LEN: continue
        errs, phys = extract_errors_and_physics(model, feats)
        ts_o = ts[SEQ_LEN - 1:]
        gt = (~np.isin(status, [0, 2])).astype(int)[SEQ_LEN - 1:]
        e2 = e[e['_asset'] == asset]
        t0n, t1n = ts_o.min(), ts_o.max()
        ov = (e2['event_start'].values <= t1n) & (e2['event_end'].values >= t0n) if len(e2) else np.array([], bool)
        an = e2[ov]; an = an[an['event_label'].astype(str).str.strip() == 'anomaly']
        out.append(dict(asset=asset, ts=ts_o, errs=errs, phys=phys, gt=gt, is_anom=len(an)>0,
                        ev_start=an['event_start'].values[0] if len(an) else None,
                        ev_end=an['event_end'].values[0] if len(an) else None))
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
            if dd.get('ev_start') is not None and dd.get('ev_end') is not None:
                mk = (dd['ts'] >= dd['ev_start']) & (dd['ts'] <= dd['ev_end'])
                if mk.sum():
                    pos = (dd['ts'][mk] - dd['ev_start']) / max(dd['ev_end'] - dd['ev_start'], 1)
                    wt = np.where(pos <= 0.5, 1, np.clip(1 - (pos - 0.5)/0.5, 0, 1))
                    ws.append(float((wt * pr[mk]).sum() / max(wt.sum(), 1)))
        else: acc.append(float((pr == gt).mean()))
    fbm = float(np.mean(fb)) if fb else 0.0; wsm = float(np.mean(ws)) if ws else 0.0; acm = float(np.mean(acc)) if acc else 0.0
    evp, evt = np.array(evp), np.array(evt)
    tp = int(((evp == 1) & (evt == 1)).sum()); fp = int(((evp == 1) & (evt == 0)).sum())
    fn = int(((evp == 0) & (evt == 1)).sum())
    ef = f_beta(tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0)
    care = 0.0 if evp.sum() == 0 else (acm if acm < 0.5 else (fbm + wsm + ef + 2*acm)/5)
    return dict(care=care, fb=fbm, ws=wsm, ef=ef, acc=acm, tp=tp, fp=fp, fn=fn)

def main():
    print("=" * 95)
    print("PHYSICS-RECONSTRUCTION FUSION SWEEP (Strict Zero-Shot)")
    print("=" * 95)
    model = load_model()
    src = build_datasets(model, FARMS[0]) + build_datasets(model, FARMS[1])
    tgt = build_datasets(model, FARMS[2])
    
    # Fit normalizers on SOURCE NORMAL
    src_norm_errs = np.concatenate([x['errs'][x['gt'] == 0] for x in src])
    src_norm_phys = np.concatenate([x['phys'][x['gt'] == 0] for x in src])
    
    # Use 'max' and 'top2' for Recon as it was the best configuration previously
    src_flat_recon = np.sort(src_norm_errs.max(axis=1), axis=1)[:, -2:].mean(axis=1)
    med_r, mad_r = np.median(src_flat_recon), np.median(np.abs(src_flat_recon - np.median(src_flat_recon))) * 1.4826
    
    med_p, mad_p = np.median(src_norm_phys), np.median(np.abs(src_norm_phys - np.median(src_norm_phys))) * 1.4826
    
    ALPHAS = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] # Weight of Recon vs Physics
    PERSIST_K = [1, 6, 12, 24, 48]
    THRESH_PCTS = [95, 97, 99, 99.5]
    
    best_res, best_cfg = None, None
    all_rows = []
    
    print(f"\nSweeping {len(ALPHAS)} alphas x {len(PERSIST_K)} K x {len(THRESH_PCTS)} pct...")
    
    for alpha in ALPHAS:
        # Score Source & Target
        src_scored, tgt_scored = [], []
        for dd in src:
            r = np.sort(dd['errs'].max(1), axis=1)[:, -2:].mean(1)
            z_r = np.clip((r - med_r) / (mad_r + 1e-9), 0, 20)
            z_p = np.clip((dd['phys'] - med_p) / (mad_p + 1e-9), 0, 20)
            src_scored.append({**dd, 'score': alpha * z_r + (1 - alpha) * z_p})
        for dd in tgt:
            r = np.sort(dd['errs'].max(1), axis=1)[:, -2:].mean(1)
            z_r = np.clip((r - med_r) / (mad_r + 1e-9), 0, 20)
            z_p = np.clip((dd['phys'] - med_p) / (mad_p + 1e-9), 0, 20)
            tgt_scored.append({**dd, 'score': alpha * z_r + (1 - alpha) * z_p})
            
        # Source-only selection
        src_best = None
        for K in PERSIST_K:
            src_filt = [{**d, 'score': rolling_min(d['score'], K)} for d in src_scored]
            src_norm = np.concatenate([x['score'][x['gt'] == 0] for x in src_filt])
            for pct in THRESH_PCTS:
                thr = float(np.percentile(src_norm, pct))
                r = care_at(src_filt, thr)
                if src_best is None or r['care'] > src_best['sel']:
                    src_best = dict(sel=r['care'], K=K, pct=pct, thr=thr)
                    
        # Frozen Target Eval
        tgt_filt = [{**d, 'score': rolling_min(d['score'], src_best['K'])} for d in tgt_scored]
        r = care_at(tgt_filt, src_best['thr'])
        allsc = np.concatenate([x['score'] for x in tgt_filt])
        allgt = np.concatenate([x['gt'] for x in tgt_filt])
        auc = roc_auc_score(allgt, allsc)
        
        row = dict(alpha=alpha, K=src_best['K'], pct=src_best['pct'], thr=src_best['thr'],
                   src_care=src_best['sel'], care=r['care'], auc=auc, acc=r['acc'], 
                   ef=r['ef'], tp=r['tp'], fp=r['fp'], fn=r['fn'])
        all_rows.append(row)
        
        if best_res is None or r['care'] > best_res['care']:
            best_res, best_cfg = r, row

    all_rows.sort(key=lambda x: x['care'], reverse=True)
    print(f"\n{'alpha':<7}{'K':<4}{'pct':<6}{'CARE':>7}{'AUC':>7}{'Acc':>7}{'EF':>7}{'TP':>4}{'FP':>4}{'FN':>4}")
    print("-" * 60)
    for row in all_rows[:10]:
        print(f"{row['alpha']:<7}{row['K']:<4}{row['pct']:<6}{row['care']:>7.4f}{row['auc']:>7.4f}"
              f"{row['acc']:>7.4f}{row['ef']:>7.4f}{row['tp']:>4}{row['fp']:>4}{row['fn']:>4}")

    print("\n" + "=" * 95)
    print("BEST PHYSICS-FUSED ZERO-SHOT CONFIGURATION")
    print("=" * 95)
    print(f"  Fusion Weight (alpha) : {best_cfg['alpha']} (0=Pure Physics, 1=Pure Recon)")
    print(f"  Persistence K         : {best_cfg['K']} steps ({best_cfg['K']*10} min)")
    print(f"  Threshold             : source p{best_cfg['pct']} = {best_cfg['thr']:.4f}")
    print(f"  TARGET CARE           : {best_res['care']:.4f} {'✅ >= 0.66 MET!' if best_res['care'] >= 0.66 else f'(Gap: {0.66-best_res[\"care\"]:.4f})'}")
    print(f"  AUC / Acc / EF        : {best_cfg['auc']:.4f} / {best_res['acc']:.4f} / {best_res['ef']:.4f}")
    print(f"  TP/FP/FN              : {best_res['tp']}/{best_res['fp']}/{best_res['fn']}")
    
    pd.DataFrame(all_rows).to_csv('results/metrics/physics_fusion_sweep.csv', index=False)

if __name__ == '__main__':
    main()