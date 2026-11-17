import os, time, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import roc_auc_score, precision_score, recall_score

torch.set_float32_matmul_precision('high')
pl.seed_everything(42)

SEQ_LEN = 36
BETA = 0.5
N_FEAT = 12
CORE8 = list(range(8))
AERO_IDX = [0, 1, 2, 3, 4, 8]          # wind, rotor, power, pitch, ambient, power_residual
THERM_IDX = [5, 6, 7, 9, 10, 11]        # oil_rise, brg_max_rise, brg_spread, electrical, hydraulic, hyd_mask
BASE_DIR = "data/raw/CARE_To_Compare"
FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CKPT_DIR = "ckpt_strict_zeroshot"
TEMPORAL_CANDIDATES = [('none', 0), ('rmin', 6), ('rmin', 12), ('rmin', 24), ('rmin', 48)]
THRESH_PCTS = [95, 97, 99, 99.5]
INPUT_MODES = ['full12', 'zeropad8']

# ============================================================
# MODEL  (self-contained; identical dual-stream physics design)
# ============================================================
class DualStreamZS(pl.LightningModule):
    def __init__(self, d_model=64, latent_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.aero_enc = nn.LSTM(len(AERO_IDX), d_model, batch_first=True, bidirectional=True)
        self.therm_enc = nn.LSTM(len(THERM_IDX), d_model, batch_first=True, bidirectional=True)
        self.fusion = nn.Sequential(nn.Linear(4 * d_model, 128), nn.ReLU(), nn.Linear(128, latent_dim))
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256), nn.ReLU(),
            nn.Linear(256, 512), nn.ReLU(),
            nn.Linear(512, SEQ_LEN * len(CORE8))
        )
        # physics params (softplus=>positive gain, sigmoid=>decay in (0,1))
        self.alpha_oil = nn.Parameter(torch.tensor(-2.0))
        self.beta_oil = nn.Parameter(torch.tensor(-1.0))
        self.alpha_brg = nn.Parameter(torch.tensor(-2.0))
        self.beta_brg = nn.Parameter(torch.tensor(-1.0))
        self.lr = lr

    def encode(self, x):
        a, _ = self.aero_enc(x[:, :, AERO_IDX]); a = a[:, -1, :]
        t, _ = self.therm_enc(x[:, :, THERM_IDX]); t = t[:, -1, :]
        return self.fusion(torch.cat([a, t], -1))

    def forward(self, x):
        z = self.encode(x)
        dec = self.decoder(z).view(-1, SEQ_LEN, len(CORE8))
        return dec, z

    def physics_loss(self, x):
        P, oil, brg = x[:, :, 2], x[:, :, 5], x[:, :, 6]
        ao, bo = F.softplus(self.alpha_oil), torch.sigmoid(self.beta_oil)
        ab, bb = F.softplus(self.alpha_brg), torch.sigmoid(self.beta_brg)
        oil_hat = (1 - bo) * oil[:, :-1] + ao * P[:, :-1]
        brg_hat = (1 - bb) * brg[:, :-1] + ab * P[:, :-1]
        return F.mse_loss(oil_hat, oil[:, 1:]) + F.mse_loss(brg_hat, brg[:, 1:])

    def training_step(self, batch, bidx):
        x, fid = batch
        dec, z = self(x)
        rec = F.mse_loss(dec, x[:, :, CORE8])
        phys = self.physics_loss(x)
        z0, z1 = z[fid == 0], z[fid == 1]
        align = torch.tensor(0.0, device=x.device)
        if len(z0) > 16 and len(z1) > 16:
            align = F.mse_loss(z0.mean(0), z1.mean(0)) + F.mse_loss(z0.std(0), z1.std(0))
        loss = rec + 0.5 * phys + 0.2 * align
        self.log('train_loss', loss, prog_bar=True)
        self.log('rec', rec); self.log('phys', phys)
        return loss

    def validation_step(self, batch, bidx):
        x, fid = batch
        dec, z = self(x)
        self.log('val_loss', F.mse_loss(dec, x[:, :, CORE8]), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

# ============================================================
# DATA  (strict: Farm A+B only)
# ============================================================
def np_path(name):
    p = os.path.join('data/processed', name)
    if not os.path.exists(p):
        p = name
    if not os.path.exists(p):
        raise FileNotFoundError(name)
    return p

def load_train_AB():
    X = np.load(np_path('v5_train_X.npy'))
    Flab = np.load(np_path('v5_train_F.npy'))
    total = len(Flab)
    mask = np.isin(Flab, [0, 1])          # <<< Farm A(0)+B(1) ONLY, Farm C(2) EXCLUDED
    X, Flab = X[mask], Flab[mask]
    print(f"STRICT ZERO-SHOT: total windows={total}, Farm A+B used={len(X)}, "
          f"Farm C EXCLUDED={total - len(X)}")
    return X, Flab

def windows(feats, stride=1):
    T = len(feats)
    if T < SEQ_LEN:
        return np.empty((0, SEQ_LEN, feats.shape[1]), dtype=np.float32)
    w = np.lib.stride_tricks.sliding_window_view(feats, window_shape=SEQ_LEN, axis=0)
    return np.ascontiguousarray(w[::stride].transpose(0, 2, 1), dtype=np.float32)

def f_beta(p, r, beta=BETA):
    d = (beta ** 2) * p + r
    return (1 + beta ** 2) * p * r / d if d > 0 else 0.0

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

# ============================================================
# TRAIN (Farm A+B only)
# ============================================================
def train_model():
    X, Flab = load_train_AB()
    n = len(X)
    idx = np.random.RandomState(42).permutation(n)
    nval = int(0.1 * n)
    val_idx, tr_idx = idx[:nval], idx[nval:]
    Xtr = torch.tensor(X[tr_idx]).float(); Ftr = torch.tensor(Flab[tr_idx]).long()
    Xva = torch.tensor(X[val_idx]).float(); Fva = torch.tensor(Flab[val_idx]).long()
    tr_dl = DataLoader(TensorDataset(Xtr, Ftr), batch_size=2048, shuffle=True, num_workers=0)
    va_dl = DataLoader(TensorDataset(Xva, Fva), batch_size=2048, num_workers=0)

    model = DualStreamZS()
    ckpt_cb = ModelCheckpoint(dirpath=CKPT_DIR, filename='best', monitor='val_loss', mode='min', save_top_k=1)
    es = EarlyStopping(monitor='val_loss', patience=6, mode='min')
    trainer = pl.Trainer(
        accelerator='gpu' if DEVICE.type == 'cuda' else 'cpu', devices=1,
        max_epochs=30, precision=32, gradient_clip_val=1.0,
        log_every_n_steps=20, callbacks=[ckpt_cb, es], enable_progress_bar=True)
    trainer.fit(model, tr_dl, va_dl)

    best = DualStreamZS.load_from_checkpoint(ckpt_cb.best_model_path).to(DEVICE).eval()
    bo = torch.sigmoid(best.beta_oil).item(); bb = torch.sigmoid(best.beta_brg).item()
    print(f"Learned physics -> oil tau={ (1/bo)*10 :.0f} min, brg tau={ (1/bb)*10 :.0f} min")
    return best

# ============================================================
# SCORING + EVAL (identical protocol to frozen evaluation)
# ============================================================
def compute_core8_errors(model, feats, input_mode):
    w = windows(feats, 1)
    if len(w) == 0:
        return np.empty((0, 8), dtype=np.float32)
    xt = torch.from_numpy(w).to(DEVICE).float()
    if input_mode == 'zeropad8':
        xt = torch.cat([xt[:, :, :8], torch.zeros_like(xt[:, :, 8:])], -1)
    out = []
    with torch.no_grad():
        for i in range(0, len(xt), 2048):
            xb = xt[i:i + 2048]
            dec, _ = model(xb)
            sq = (dec - xb[:, :, :8]) ** 2
            out.append(sq[:, -6:, :].mean(1).cpu().numpy())
    return np.concatenate(out).astype(np.float32)

def pred_path(farm):
    p = f'data/processed/v5_pred_{farm.replace(" ", "_")}.npy'
    if not os.path.exists(p):
        p = f'v5_pred_{farm.replace(" ", "_")}.npy'
    return p

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
            an = e2[ov]; an = an[an['event_label'].astype(str).str.strip() == 'anomaly']
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
        s = z.mean(1)
        s = temporal_filter(s, temp_spec)
        res.append({**dd, 'score': s})
    return res

def main():
    t0 = time.time()
    print("=" * 95)
    print("STRICT ZERO-SHOT: train on Farm A+B only  ->  select on A+B  ->  freeze on Farm C")
    print("=" * 95)
    print(f"DEVICE: {DEVICE}")
    model = train_model()

    results_by_mode = {}
    for input_mode in INPUT_MODES:
        print("\n" + "=" * 95)
        print(f"HYDRAULIC ABLATION ARM: input_mode = {input_mode}")
        print("=" * 95)
        src = build_datasets(model, FARMS[0], input_mode) + build_datasets(model, FARMS[1], input_mode)
        tgt = build_datasets(model, FARMS[2], input_mode)
        print(f"  Source(A+B) datasets={len(src)} (anomalous={sum(x['is_anom'] for x in src)})  "
              f"Target(C) datasets={len(tgt)} (anomalous={sum(x['is_anom'] for x in tgt)})")

        src_norm = np.concatenate([x['errs'][x['gt'] == 0] for x in src])
        med = np.median(src_norm, axis=0)
        mad = np.median(np.abs(src_norm - med), axis=0) * 1.4826
        print(f"  Robust normalizer fitted on {len(src_norm)} source-normal windows.")

        print("  Selecting config + threshold on SOURCE (A+B) only ...")
        best = None
        for temp_spec in TEMPORAL_CANDIDATES:
            src_scored = apply_pipeline(src, med, mad, temp_spec)
            src_normal_scores = np.concatenate([x['score'][x['gt'] == 0] for x in src_scored])
            for pct in THRESH_PCTS:
                thr = float(np.percentile(src_normal_scores, pct))
                r = care_at(src_scored, thr)
                if best is None or r['care'] > best['sel']:
                    best = dict(sel=r['care'], temp=temp_spec, pct=pct, thr=thr, src_care=r['care'])
        print(f"    SELECTED (source-only): temporal={best['temp']} thr_pct={best['pct']} "
              f"thr={best['thr']:.4f} source_CARE={best['src_care']:.4f}")

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

        print(f"\n  FROZEN FARM C RESULT ({input_mode})  [model NEVER trained on C]:")
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

    print("\n" + "=" * 95)
    print("STRICT ZERO-SHOT SUMMARY (train A+B -> test C)")
    print("=" * 95)
    print(f"{'input':<11}{'CARE':>7}{'AUC':>7}{'Prec':>7}{'Rec':>7}{'EF':>7}{'Acc':>7}"
          f"{'TP':>4}{'FP':>4}{'FN':>4}{'FA-ds':>8}{'sel_temp':>14}{'sel_pct':>9}")
    for im in INPUT_MODES:
        m = results_by_mode[im]
        print(f"{im:<11}{m['care']:>7.4f}{m['auc']:>7.4f}{m['prec']:>7.4f}{m['rec']:>7.4f}"
              f"{m['ef']:>7.4f}{m['acc']:>7.4f}{m['tp']:>4}{m['fp']:>4}{m['fn']:>4}"
              f"{str(m['fa_datasets'])+'/'+str(m['n_norm_ds']):>8}{m['selected_temp']:>14}{m['selected_pct']:>9}")

    os.makedirs('results/metrics', exist_ok=True)
    with open('results/metrics/strict_zeroshot_eval.json', 'w') as f:
        json.dump(results_by_mode, f, indent=2)
    print("\nSaved: results/metrics/strict_zeroshot_eval.json")
    print(f"Elapsed: {time.time()-t0:.0f}s")
    print("PROTOCOL: trained ONLY on Farm A+B. Farm C excluded from training AND selection. Evaluated once.")

if __name__ == '__main__':
    main()