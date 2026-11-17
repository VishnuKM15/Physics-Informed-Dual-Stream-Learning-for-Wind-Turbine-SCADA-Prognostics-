import os, glob, time
import numpy as np
import pandas as pd
import torch, torch.nn.functional as F
from sklearn.metrics import roc_auc_score, precision_score, recall_score

SEQ_LEN = 36; BETA = 0.5
BASE_DIR = "data/raw/CARE_To_Compare"
FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Reconstruction weights used in v5 training (must match train_v5.py)
W12 = torch.tensor([1,1,1,1,1,3,3,3,1,2,2,0], dtype=torch.float32)

def windows(feats, stride=1):
    T = len(feats)
    if T < SEQ_LEN: return np.empty((0, SEQ_LEN, feats.shape[1]), dtype=np.float32)
    w = np.lib.stride_tricks.sliding_window_view(feats, window_shape=SEQ_LEN, axis=0)
    return np.ascontiguousarray(w[::stride].transpose(0,2,1), dtype=np.float32)

def f_beta(p, r, beta=BETA):
    return (1+beta**2)*p*r/(beta**2*p+r) if (beta**2*p+r) else 0.0

def load_model():
    # Import the model class from the training script so the checkpoint matches
    import importlib.util, sys
    spec = importlib.util.spec_from_file_location("train_v5", "train_v5.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    ckpts = sorted(glob.glob("ckpt5/*.ckpt"), key=os.path.getmtime)
    if not ckpts:
        raise FileNotFoundError("No checkpoint in ckpt5/. Did training save one?")
    ckpt = ckpts[-1]
    print(f"Loading checkpoint: {ckpt}")
    model = mod.DualStreamV5.load_from_checkpoint(ckpt, map_location=DEVICE).to(DEVICE).eval()
    return model

def per_feature_recon(model, feats_tensor):
    """Return per-feature mean squared error across windows."""
    with torch.no_grad():
        xr, _ = model(feats_tensor)
        sq = (xr - feats_tensor) ** 2           # (B, T, 12)
        return sq.mean(dim=1).mean(dim=0).cpu().numpy()  # (12,)

def build_datasets():
    ev = {}
    for f in FARMS:
        e = pd.read_csv(os.path.join(BASE_DIR, f, 'event_info.csv'), sep=None, engine='python')
        e.columns = e.columns.str.strip()
        e['event_start'] = pd.to_datetime(e['event_start']).astype('int64')
        e['event_end'] = pd.to_datetime(e['event_end']).astype('int64')
        ev[f] = e
    datasets = {f: [] for f in FARMS}
    for farm in FARMS:
        d = np.load(f'v5_pred_{farm.replace(" ", "_")}.npy', allow_pickle=True).item()
        for j in np.unique(d['ds']):
            m = d['ds'] == j
            feats, ts, status = d['feats'][m], d['ts'][m], d['status'][m]
            if len(feats) < SEQ_LEN: continue
            dd = dict(feats=feats, ts=ts, status=status)
            datasets[farm].append(dd)
    # attach events
    for f in FARMS:
        edf = ev[f]; ac = 'asset' if 'asset' in edf.columns else 'asset_id'
        # asset stored per-row in v5
        d = np.load(f'v5_pred_{f.replace(" ", "_")}.npy', allow_pickle=True).item()
        assets = d['asset']; ds = d['ds']
        for dd in datasets[f]:
            j = dd.get('_j')
        # simpler: re-derive asset per dataset by re-looping
    return datasets, ev

def main():
    t0 = time.time()
    print("="*90); print("V5 DIAGNOSIS: feature-subset scoring"); print("="*90)
    print(f"DEVICE: {DEVICE}")
    model = load_model()

    ev = {}
    for f in FARMS:
        e = pd.read_csv(os.path.join(BASE_DIR, f, 'event_info.csv'), sep=None, engine='python')
        e.columns = e.columns.str.strip()
        e['event_start'] = pd.to_datetime(e['event_start']).astype('int64')
        e['event_end'] = pd.to_datetime(e['event_end']).astype('int64')
        ev[f] = e

    # ---- 1. Per-feature recon error: A+B normal vs C normal vs C anomaly ----
    # gather windows
    def collect(farm, only_status=None):
        d = np.load(f'v5_pred_{farm.replace(" ", "_")}.npy', allow_pickle=True).item()
        out = []
        for j in np.unique(d['ds']):
            m = d['ds'] == j
            feats = d['feats'][m]; status = d['status'][m]
            if only_status is not None:
                kk = np.isin(status, only_status)
                feats = feats[kk]
            if len(feats) >= SEQ_LEN:
                out.append(windows(feats, SEQ_LEN//2))
        return np.concatenate(out) if out else np.empty((0,SEQ_LEN,12))

    src_normal = np.concatenate([collect(f,[0,2]) for f in FARMS[:2]])
    c_normal = collect(FARMS[2],[0,2])
    c_abnormal = collect(FARMS[2],[1,3,4,5])

    print("\n[1] Per-feature reconstruction MSE (lower=cleaner reconstruction)")
    fnames = ['wind','rotor','power','pitch','amb','oil_rise','brg_rise','brg_spr',
              'pwr_res','electr','hydraul','hyd_mask']
    rows = {}
    for label, arr in [('A+B normal', src_normal[:4000]),
                       ('C normal',   c_normal[:4000]),
                       ('C abnormal', c_abnormal[:4000])]:
        if len(arr)==0: continue
        xt = torch.from_numpy(arr).to(DEVICE)
        pf = []
        for i in range(0, len(xt), 2048):
            pf.append(per_feature_recon(model, xt[i:i+2048]))
        rows[label] = np.mean(pf, axis=0)

    print(f"{'feature':<10}", *[f"{k[:10]:>12}" for k in rows])
    for i, n in enumerate(fnames):
        print(f"{n:<10}", *[f"{rows[k][i]:>12.5f}" for k in rows])

    # ---- 2. Score with FULL 12 vs CORE 8 (exclude electr/hydraul/hyd_mask) ----
    def score_windows(arr, use_core8):
        xt = torch.from_numpy(arr).to(DEVICE)
        scores = []
        with torch.no_grad():
            for i in range(0, len(xt), 2048):
                xb = xt[i:i+2048].float()
                xr, _ = model(xb)
                sq = (xr - xb) ** 2
                if use_core8:
                    w = W12.clone(); w[8:12] = 0   # zero out new features
                else:
                    w = W12
                w = w.to(xb.device)
                scores.append((sq * w).sum(-1).mean(-1).cpu().numpy())
        return np.concatenate(scores)

    def care_from_scores(ds_list, thr):
        fb, ws, acc, evp, evt = [], [], [], [], []
        for dd in ds_list:
            pr = (dd['score'] > thr).astype(int); gt = dd['gt']
            evp.append(int(pr.sum()>0)); evt.append(int(dd['is_anom']))
            if dd['is_anom']:
                tp=int(((pr==1)&(gt==1)).sum()); fp=int(((pr==1)&(gt==0)).sum()); fn=int(((pr==0)&(gt==1)).sum())
                fb.append(f_beta(tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0))
                if dd.get('ev_start') is not None:
                    mk=(dd['ts']>=dd['ev_start'])&(dd['ts']<=dd['ev_end'])
                    if mk.sum():
                        pos=(dd['ts'][mk]-dd['ev_start'])/max(dd['ev_end']-dd['ev_start'],1)
                        wts=np.where(pos<=0.5,1,np.clip(1-(pos-0.5)/0.5,0,1))
                        ws.append(float((wts*pr[mk]).sum()/max(wts.sum(),1)))
            else:
                acc.append(float((pr==gt).mean()))
        fbm=np.mean(fb) if fb else 0; wsm=np.mean(ws) if ws else 0; acm=np.mean(acc) if acc else 0
        evp=np.array(evp); evt=np.array(evt)
        tp=int(((evp==1)&(evt==1)).sum()); fp=int(((evp==1)&(evt==0)).sum()); fn=int(((evp==0)&(evt==1)).sum())
        ef=f_beta(tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0)
        care = 0 if evp.sum()==0 else (acm if acm<0.5 else (fbm+wsm+ef+2*acm)/5)
        return dict(care=care, fb=fbm, ws=wsm, ef=ef, acc=acm)

    def build_scored(farm, use_core8):
        d = np.load(f'v5_pred_{farm.replace(" ", "_")}.npy', allow_pickle=True).item()
        edf = ev[farm]; ac = 'asset' if 'asset' in edf.columns else 'asset_id'
        out = []
        for j in np.unique(d['ds']):
            m = d['ds'] == j
            feats, ts, status = d['feats'][m], d['ts'][m], d['status'][m]
            asset = str(d['asset'][m][0]).strip()
            if len(feats) < SEQ_LEN: continue
            w = windows(feats, 1)
            sc = score_windows(w, use_core8)
            gt = (~np.isin(status,[0,2])).astype(int)[SEQ_LEN-1:]
            e2 = edf[edf[ac]==asset]
            t0n, t1n = ts[SEQ_LEN-1:].min(), ts[SEQ_LEN-1:].max()
            ov = (e2['event_start'].values<=t1n)&(e2['event_end'].values>=t0n) if len(e2) else np.array([],bool)
            anom = e2[ov]; anom = anom[anom['event_label']=='anomaly']
            out.append(dict(ts=ts[SEQ_LEN-1:], score=sc, gt=gt, is_anom=len(anom)>0,
                            ev_start=anom['event_start'].values[0] if len(anom) else None,
                            ev_end=anom['event_end'].values[0] if len(anom) else None))
        return out

    print("\n[2] Scoring subsets (calibrate on A+B, test on C)")
    for use_core8, label in [(False,'FULL 12 features'), (True,'CORE 8 (drop pwr_res/electr/hydraul/mask)')]:
        src_ds = build_scored(FARMS[0], use_core8) + build_scored(FARMS[1], use_core8)
        cal = np.concatenate([dd['score'][dd['gt']==0] for dd in src_ds])
        best, bthr = -1, None
        for p in [95,97,99,99.5,99.9]:
            thr = float(np.percentile(cal,p))
            r = care_from_scores(src_ds, thr)
            if r['ef']+r['fb'] > best: best, bthr = r['ef']+r['fb'], thr
        test_ds = build_scored(FARMS[2], use_core8)
        r = care_from_scores(test_ds, bthr)
        allsc = np.concatenate([dd['score'] for dd in test_ds])
        allgt = np.concatenate([dd['gt'] for dd in test_ds])
        prc = (allsc>bthr).astype(int)
        auc = roc_auc_score(allgt, allsc)
        print(f"\n  {label}")
        print(f"    CARE={r['care']:.4f} fb={r['fb']:.3f} ws={r['ws']:.3f} ef={r['ef']:.3f} acc={r['acc']:.3f}")
        print(f"    AUC={auc:.4f} P={precision_score(allgt,prc,zero_division=0):.3f} R={recall_score(allgt,prc,zero_division=0):.3f}")

    print(f"\nDone in {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()