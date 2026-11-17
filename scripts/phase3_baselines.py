import os, glob, json, time
import numpy as np, pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, precision_score, recall_score

torch.set_float32_matmul_precision('high')
SEQ_LEN = 36; BETA = 0.5
BASE_DIR = "data/raw/CARE_To_Compare"
FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"DEVICE: {DEVICE}")

def windows(feats, stride):
    T = len(feats)
    if T < SEQ_LEN: return np.empty((0, SEQ_LEN, feats.shape[1]), dtype=np.float32)
    w = np.lib.stride_tricks.sliding_window_view(feats, window_shape=SEQ_LEN, axis=0)
    return np.ascontiguousarray(w[::stride].transpose(0, 2, 1), dtype=np.float32)

def f_beta(p, r, beta=BETA):
    return (1+beta**2)*p*r/(beta**2*p+r) if (beta**2*p+r) else 0.0

def build_datasets():
    ev = {}
    for f in FARMS:
        e = pd.read_csv(os.path.join(BASE_DIR, f, "event_info.csv"), sep=None, engine="python")
        e.columns = e.columns.str.strip()
        e["event_start"] = pd.to_datetime(e["event_start"]).astype("int64")
        e["event_end"] = pd.to_datetime(e["event_end"]).astype("int64")
        ev[f] = e
    datasets = {f: [] for f in FARMS}
    for fi, farm in enumerate(FARMS):
        d = np.load(f'v3_pred_{farm.replace(" ", "_")}.npy', allow_pickle=True).item()
        for j in np.unique(d["ds"]):
            m = d["ds"] == j
            feats, ts, status = d["feats"][m], d["ts"][m], d["status"][m]
            if len(feats) < SEQ_LEN: continue
            datasets[farm].append(dict(
                feats=feats, ts=ts[SEQ_LEN-1:], status=status,
                gt=(~np.isin(status, [0, 2])).astype(int)[SEQ_LEN-1:],
                asset=d["asset"][m][0]))
    for f in FARMS:
        edf = ev[f]; ac = "asset" if "asset" in edf.columns else "asset_id"
        for d in datasets[f]:
            e2 = edf[edf[ac] == d["asset"]]
            t0n, t1n = d["ts"].min(), d["ts"].max()
            ov = (e2["event_start"].values <= t1n) & (e2["event_end"].values >= t0n) if len(e2) else np.array([], bool)
            d["is_anom"] = bool(len(e2) and (e2[ov]["event_label"] == "anomaly").any())
            d["ev_start"] = e2[ov]["event_start"].values[0] if d["is_anom"] else None
            d["ev_end"] = e2[ov]["event_end"].values[0] if d["is_anom"] else None
    return datasets

def care(ds_list, thr):
    fb, ws, acc, evp, evt, ear = [], [], [], [], [], []
    for d in ds_list:
        pr = (d["scores"] > thr).astype(int); gt = d["gt"]
        evp.append(int(pr.sum() > 0)); evt.append(int(d["is_anom"]))
        if d["is_anom"]:
            tp = int(((pr==1)&(gt==1)).sum()); fp = int(((pr==1)&(gt==0)).sum()); fn = int(((pr==0)&(gt==1)).sum())
            fb.append(f_beta(tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0))
            if d["ev_start"] is not None:
                mask = (d["ts"] >= d["ev_start"]) & (d["ts"] <= d["ev_end"])
                if mask.sum():
                    pos = (d["ts"][mask]-d["ev_start"]) / max(d["ev_end"]-d["ev_start"], 1)
                    wts = np.where(pos <= 0.5, 1, np.clip(1-(pos-0.5)/0.5, 0, 1))
                    ws.append(float((wts*pr[mask]).sum()/wts.sum()))
                i0 = int(np.argmax(pr==1)) if pr.sum() else None
                if i0 is not None and d["ts"][i0] < d["ev_start"]:
                    ear.append(float((d["ev_start"]-d["ts"][i0])/86400e9))
        else:
            acc.append(float((pr==gt).mean()))
    fbm = float(np.mean(fb)) if fb else 0.0; wsm = float(np.mean(ws)) if ws else 0.0; acm = float(np.mean(acc)) if acc else 0.0
    evp = np.array(evp); evt = np.array(evt)
    tp = int(((evp==1)&(evt==1)).sum()); fp = int(((evp==1)&(evt==0)).sum()); fn = int(((evp==0)&(evt==1)).sum())
    ef = f_beta(tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0)
    if evp.sum() == 0: cs = 0.0
    elif acm < 0.5: cs = acm
    else: cs = (fbm+wsm+ef+2*acm)/5
    return dict(care=cs, fb=fbm, ws=wsm, ef=ef, acc=acm, ear=ear)

def evaluate_protocol(datasets, score_fn, name):
    for farm in FARMS:
        for d in datasets[farm]:
            d["scores"] = score_fn(d)
    src = datasets[FARMS[0]] + datasets[FARMS[1]]
    normal = np.concatenate([d["scores"][d["gt"] == 0] for d in src])
    best = -1; bthr = None
    for p in [95, 97, 99, 99.5, 99.9]:
        thr = float(np.percentile(normal, p))
        r = care(src, thr)
        if r["ef"] + r["fb"] > best: best, bthr = r["ef"]+r["fb"], thr
    r = care(datasets[FARMS[2]], bthr)
    allsc = np.concatenate([d["scores"] for d in datasets[FARMS[2]]])
    allgt = np.concatenate([d["gt"] for d in datasets[FARMS[2]]])
    prc = (allsc > bthr).astype(int)
    auc = roc_auc_score(allgt, allsc)
    prec = precision_score(allgt, prc, zero_division=0)
    rec = recall_score(allgt, prc, zero_division=0)
    print(f"\n{name}")
    print(f"  CARE={r['care']:.4f} (fb={r['fb']:.3f} ws={r['ws']:.3f} ef={r['ef']:.3f} acc={r['acc']:.3f})")
    print(f"  AUC={auc:.4f} Precision={prec:.3f} Recall={rec:.3f}")
    return dict(name=name, care=r['care'], auc=auc, prec=prec, rec=rec, r=r)

def main():
    t0 = time.time()
    datasets = build_datasets()

    # Source normal windows (A+B) for fitting baselines
    trX = np.load("v3_train_X.npy"); trF = np.load("v3_train_F.npy")
    src_windows = trX[np.isin(trF, [0, 1])]
    src_flat = src_windows.reshape(len(src_windows), -1).astype(np.float32)
    print(f"Source normal windows: {src_windows.shape}")

    results = []

    # ---- Baseline 1: Isolation Forest ----
    print("\nFitting Isolation Forest...")
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(src_flat)
    def iso_score(d):
        w = windows(d["feats"], 1)
        return -iso.score_samples(w.reshape(len(w), -1))
    results.append(evaluate_protocol(datasets, iso_score, "Baseline 1: Isolation Forest"))

    # ---- Baseline 2: One-Class SVM (subsampled) ----
    print("\nFitting One-Class SVM (subsampled)...")
    n = min(15000, len(src_flat))
    idx = np.random.default_rng(0).choice(len(src_flat), n, replace=False)
    oc = OneClassSVM(gamma="scale", nu=0.05)
    oc.fit(src_flat[idx])
    def oc_score(d):
        w = windows(d["feats"], 1)
        return -oc.decision_function(w.reshape(len(w), -1))
    results.append(evaluate_protocol(datasets, oc_score, "Baseline 2: One-Class SVM"))

    # ---- Baseline 3: Standard Autoencoder (no physics, no DA, no dual-stream) ----
    print("\nTraining Standard Autoencoder on A+B normal...")
    in_dim = SEQ_LEN * 8
    class AE(pl.LightningModule):
        def __init__(s):
            super().__init__(); s.save_hyperparameters()
            s.enc = nn.Sequential(nn.Linear(in_dim, 128), nn.ReLU(), nn.Linear(128, 32))
            s.dec = nn.Sequential(nn.Linear(32, 128), nn.ReLU(), nn.Linear(128, in_dim))
        def forward(s, x):
            return s.dec(s.enc(x.reshape(x.size(0), -1))).view_as(x)
        def training_step(s, b, i):
            x = b[0]; loss = F.mse_loss(s(x), x); s.log("tr", loss); return loss
        def validation_step(s, b, i):
            x = b[0]; s.log("vl", F.mse_loss(s(x), x))
        def configure_optimizers(s): return torch.optim.AdamW(s.parameters(), lr=2e-3)
            
    src_t = torch.from_numpy(src_windows).to(DEVICE)
    dl = DataLoader(TensorDataset(src_t), batch_size=8192, shuffle=True, num_workers=0)
    val = DataLoader(TensorDataset(torch.from_numpy(src_windows[:10000]).to(DEVICE)), batch_size=8192)
    ae = AE()
    trainer = pl.Trainer(accelerator="gpu" if DEVICE.type == "cuda" else "cpu", devices=1,
                         max_epochs=15, precision="bf16-mixed" if DEVICE.type == "cuda" else 32,
                         log_every_n_steps=20, enable_progress_bar=False, enable_model_summary=False)
    trainer.fit(ae, dl, val)
    
    # FIX: Explicitly move model to DEVICE after Lightning training finishes
    ae = ae.to(DEVICE)
    ae.eval()
    
    def ae_score(d):
        w = torch.from_numpy(windows(d["feats"], 1)).to(DEVICE)
        out = []
        with torch.no_grad():
            for i in range(0, len(w), 8192):
                xb = w[i:i+8192]; xr = ae(xb)
                out.append(F.mse_loss(xr, xb, reduction="none").mean((1, 2)).cpu().numpy())
        return np.concatenate(out)
    results.append(evaluate_protocol(datasets, ae_score, "Baseline 3: Standard Autoencoder"))

    # ---- Comparison table ----
    print("\n" + "="*100)
    print("IDENTICAL ZERO-SHOT PROTOCOL COMPARISON (Train A+B -> Calibrate A+B -> Test C)")
    print("="*100)
    print(f"{'Model':<32} {'CARE':>8} {'AUC':>8} {'Prec':>8} {'Recall':>8}")
    print("-"*70)
    # Reference: v4 full model (physics + DA)
    print(f"{'v4: Physics + Domain Adaptation':<32} {'0.5743':>8} {'0.8406':>8} {'0.471':>8} {'0.185':>8}")
    for res in results:
        print(f"{res['name']:<32} {res['care']:>8.4f} {res['auc']:>8.4f} {res['prec']:>8.3f} {res['rec']:>8.3f}")
    print("="*100)
    print(f"TOTAL TIME: {time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()