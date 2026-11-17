import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, precision_score, recall_score

torch.set_float32_matmul_precision("high")
torch.backends.cudnn.benchmark = True

SEQ_LEN = 36
BETA = 0.5
BATCH = 8192
EPOCHS = int(os.environ.get("ABLATION_EPOCHS", "10"))

FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_CSV = "ablation_results.csv"

CONFIGS = [
    (False, False, False),
    (False, True,  False),
    (False, False, True),
    (False, True,  True),
    (True,  False, False),
    (True,  True,  False),
    (True,  False, True),
    (True,  True,  True),
]

DIRECTIONS = [
    ("A+B->C", [0, 1], 2),
    ("A+C->B", [0, 2], 1),
    ("B+C->A", [1, 2], 0),
]

SEEDS = [42, 123, 777]


def find_base_dir():
    candidates = [
        "data/raw/CARE_To_Compare",
        "/teamspace/studios/this_studio/CARE_Project/data/raw/CARE_To_Compare",
        "/teamspace/studios/this_studio/instrumentation_01/data/raw/CARE_To_Compare",
    ]
    for c in candidates:
        if os.path.isdir(c):
            return c
    raise FileNotFoundError("Could not find CARE_To_Compare dataset directory.")


BASE_DIR = find_base_dir()


def invert_physics_to_raw(X):
    """
    Inverts v3 physics-normalized features back to raw-ish units.

    Handles BOTH:
      - 3D windows: (N, 36, 8)
      - 2D point arrays: (T, 8)
    """
    X = np.asarray(X, dtype=np.float32)
    raw = X.copy()

    if X.ndim == 2:
        amb = X[:, 4] * 100.0

        raw[:, 0] = X[:, 0] * 25.0          # wind speed
        raw[:, 1] = X[:, 1] * 20.0          # rotor speed
        raw[:, 2] = X[:, 2]                 # active power, already 0-1
        raw[:, 3] = X[:, 3] * 90.0          # pitch
        raw[:, 4] = amb                     # ambient temp
        raw[:, 5] = X[:, 5] * 100.0 + amb   # oil raw temp
        raw[:, 6] = X[:, 6] * 100.0 + amb   # bearing raw temp
        raw[:, 7] = X[:, 7] * 100.0 + amb   # generator proxy raw temp

    elif X.ndim == 3:
        amb = X[:, :, 4] * 100.0

        raw[:, :, 0] = X[:, :, 0] * 25.0
        raw[:, :, 1] = X[:, :, 1] * 20.0
        raw[:, :, 2] = X[:, :, 2]
        raw[:, :, 3] = X[:, :, 3] * 90.0
        raw[:, :, 4] = amb
        raw[:, :, 5] = X[:, :, 5] * 100.0 + amb
        raw[:, :, 6] = X[:, :, 6] * 100.0 + amb
        raw[:, :, 7] = X[:, :, 7] * 100.0 + amb

    else:
        raise ValueError(f"invert_physics_to_raw expects 2D or 3D input, got shape {X.shape}")

    return raw


def windows(feats, stride=1):
    T = len(feats)
    if T < SEQ_LEN:
        return np.empty((0, SEQ_LEN, feats.shape[1]), dtype=np.float32)

    w = np.lib.stride_tricks.sliding_window_view(feats, window_shape=SEQ_LEN, axis=0)
    return np.ascontiguousarray(w[::stride].transpose(0, 2, 1), dtype=np.float32)


def f_beta(p, r, beta=BETA):
    denom = (beta ** 2) * p + r
    if denom <= 0:
        return 0.0
    return (1 + beta ** 2) * p * r / denom


def care(ds_list, thr):
    fb, ws, acc, evp, evt = [], [], [], [], []

    for d in ds_list:
        pr = (d["scores"] > thr).astype(int)
        gt = d["gt"]

        evp.append(int(pr.sum() > 0))
        evt.append(int(d["is_anom"]))

        if d["is_anom"]:
            tp = int(((pr == 1) & (gt == 1)).sum())
            fp = int(((pr == 1) & (gt == 0)).sum())
            fn = int(((pr == 0) & (gt == 1)).sum())

            p = tp / (tp + fp) if tp + fp > 0 else 0.0
            r = tp / (tp + fn) if tp + fn > 0 else 0.0
            fb.append(f_beta(p, r))

            if d["ev_start"] is not None and d["ev_end"] is not None:
                mask = (d["ts"] >= d["ev_start"]) & (d["ts"] <= d["ev_end"])
                if mask.sum() > 0:
                    pos = (d["ts"][mask] - d["ev_start"]) / max(d["ev_end"] - d["ev_start"], 1)
                    wts = np.where(pos <= 0.5, 1.0, np.clip(1.0 - (pos - 0.5) / 0.5, 0.0, 1.0))
                    ws.append(float((wts * pr[mask]).sum() / max(wts.sum(), 1.0)))
        else:
            acc.append(float((pr == gt).mean()))

    fbm = float(np.mean(fb)) if fb else 0.0
    wsm = float(np.mean(ws)) if ws else 0.0
    acm = float(np.mean(acc)) if acc else 0.0

    evp = np.array(evp)
    evt = np.array(evt)

    tp = int(((evp == 1) & (evt == 1)).sum())
    fp = int(((evp == 1) & (evt == 0)).sum())
    fn = int(((evp == 0) & (evt == 1)).sum())

    p = tp / (tp + fp) if tp + fp > 0 else 0.0
    r = tp / (tp + fn) if tp + fn > 0 else 0.0
    ef = f_beta(p, r)

    if evp.sum() == 0:
        care_score = 0.0
    elif acm < 0.5:
        care_score = acm
    else:
        care_score = (fbm + wsm + ef + 2.0 * acm) / 5.0

    return dict(care=care_score, fb=fbm, ws=wsm, ef=ef, acc=acm)


class StreamEnc(nn.Module):
    def __init__(self, d, h=128, z=64):
        super().__init__()
        self.lstm = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(h * 2, z)

    def forward(self, x):
        return self.fc(self.lstm(x)[0][:, -1])


class AblationAE(pl.LightningModule):
    def __init__(self, use_mmd=False, use_coral=False, max_epochs=EPOCHS):
        super().__init__()
        self.save_hyperparameters()
        self.max_epochs = max_epochs

        self.aero = StreamEnc(5)
        self.therm = StreamEnc(3)

        self.fusion = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        self.dec = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, SEQ_LEN * 8)
        )

    def forward(self, x):
        z = self.fusion(torch.cat([self.aero(x[:, :, :5]), self.therm(x[:, :, 5:8])], dim=-1))
        xr = self.dec(z).view(-1, SEQ_LEN, 8)
        return xr, z

    def _sample(self, z, n=512):
        if z.size(0) > n:
            idx = torch.randperm(z.size(0), device=z.device)[:n]
            return z[idx]
        return z

    def mmd(self, x, y):
        x = x.float()
        y = y.float()

        xx = torch.cdist(x, x) ** 2
        yy = torch.cdist(y, y) ** 2
        xy = torch.cdist(x, y) ** 2

        kxx = sum(torch.exp(-xx / (2.0 * s2)) for s2 in (1.0, 5.0, 10.0)).mean()
        kyy = sum(torch.exp(-yy / (2.0 * s2)) for s2 in (1.0, 5.0, 10.0)).mean()
        kxy = sum(torch.exp(-xy / (2.0 * s2)) for s2 in (1.0, 5.0, 10.0)).mean()

        return kxx + kyy - 2.0 * kxy

    def coral(self, x, y):
        x = x.float()
        y = y.float()

        xc = x - x.mean(dim=0, keepdim=True)
        yc = y - y.mean(dim=0, keepdim=True)

        cx = xc.T @ xc / max(x.size(0) - 1, 1)
        cy = yc.T @ yc / max(y.size(0) - 1, 1)

        return (cx - cy).pow(2).sum() / (4.0 * x.size(1) ** 2)

    def _align(self, z, fid):
        if not (self.hparams.use_mmd or self.hparams.use_coral):
            return torch.tensor(0.0, device=z.device)

        z = z.float()
        uniq = torch.unique(fid)

        loss = 0.0
        npairs = 0

        for i in range(len(uniq)):
            for j in range(i + 1, len(uniq)):
                za = z[fid == uniq[i]]
                zb = z[fid == uniq[j]]

                if za.size(0) > 8 and zb.size(0) > 8:
                    za = self._sample(za)
                    zb = self._sample(zb)

                    term = 0.0
                    if self.hparams.use_mmd:
                        term = term + self.mmd(za, zb)
                    if self.hparams.use_coral:
                        term = term + self.coral(za, zb)

                    loss = loss + term
                    npairs += 1

        if npairs == 0:
            return torch.tensor(0.0, device=z.device)

        return loss / float(npairs)

    def training_step(self, batch, batch_idx):
        x, fid = batch
        x = x.float()

        xr, z = self(x)
        loss = F.mse_loss(xr, x)

        if self.hparams.use_mmd or self.hparams.use_coral:
            p = float(batch_idx + self.current_epoch * self.trainer.num_training_batches) / \
                float(max(self.max_epochs * self.trainer.num_training_batches, 1))
            lam = min(1.0, 2.0 * p)
            loss = loss + lam * self._align(z, fid)

        self.log("train_loss", loss, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        x, fid = batch
        x = x.float()

        xr, z = self(x)
        loss = F.mse_loss(xr, x)

        self.log("val_loss", loss, prog_bar=False)

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-3, weight_decay=1e-4)


def load_results():
    cols = [
        "config", "phys", "mmd", "coral", "direction", "seed",
        "CARE", "AUC", "Recall", "Precision",
        "Fbeta", "WS", "EFbeta", "Acc", "DomainAcc", "threshold", "time_s"
    ]

    if os.path.exists(RESULTS_CSV):
        try:
            return pd.read_csv(RESULTS_CSV)
        except Exception:
            pass

    return pd.DataFrame(columns=cols)


def load_event_info():
    ev = {}

    for f in FARMS:
        e = pd.read_csv(os.path.join(BASE_DIR, f, "event_info.csv"), sep=None, engine="python")
        e.columns = e.columns.str.strip()

        ac = "asset" if "asset" in e.columns else "asset_id"
        e["_asset_key"] = e[ac].astype(str).str.strip()

        e["event_start"] = pd.to_datetime(e["event_start"]).values.astype("int64")
        e["event_end"] = pd.to_datetime(e["event_end"]).values.astype("int64")

        ev[f] = e

    return ev


def load_pred_phys():
    required = []
    out = {}

    for fi, f in enumerate(FARMS):
        fname = f"v3_pred_{f.replace(' ', '_')}.npy"
        required.append(fname)

        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing required file: {fname}")

        out[fi] = np.load(fname, allow_pickle=True).item()

    return out


def build_direction(src, tgt, trX_phys, trF, pred_phys):
    src_mask = np.isin(trF, src)

    phys_train = trX_phys[src_mask]
    raw_train = invert_physics_to_raw(phys_train)

    sc = StandardScaler()
    sc.fit(raw_train.reshape(-1, 8))

    std_train = sc.transform(raw_train.reshape(-1, 8)).reshape(raw_train.shape).astype(np.float32)
    train_fid = trF[src_mask]

    pred_std = {}
    for fi in range(3):
        raw = invert_physics_to_raw(pred_phys[fi]["feats"])
        feats = sc.transform(raw).astype(np.float32)

        pred_std[fi] = dict(pred_phys[fi])
        pred_std[fi]["feats"] = feats

    return dict(
        phys_train=phys_train,
        std_train=std_train,
        train_fid=train_fid,
        pred_phys=pred_phys,
        pred_std=pred_std,
    )


def train_model(train_X, train_F, use_mmd, use_coral, seed):
    pl.seed_everything(seed, workers=True)

    N = train_X.size(0)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(N)

    val_n = min(20000, max(1000, N // 10))
    val_idx = idx[:val_n]
    tr_idx = idx[val_n:]

    xtr = train_X[tr_idx]
    ftr = train_F[tr_idx]

    xval = train_X[val_idx]
    fval = train_F[val_idx]

    dl = DataLoader(TensorDataset(xtr, ftr), batch_size=BATCH, shuffle=True, num_workers=0)
    vl = DataLoader(TensorDataset(xval, fval), batch_size=BATCH, shuffle=False, num_workers=0)

    model = AblationAE(use_mmd=use_mmd, use_coral=use_coral, max_epochs=EPOCHS)

    ckpt_cb = ModelCheckpoint(
        dirpath="ckpt_ablation",
        filename=f"run_{int(time.time() * 1000)}",
        monitor="val_loss",
        mode="min",
        save_top_k=1
    )

    trainer = pl.Trainer(
        accelerator="gpu" if DEVICE.type == "cuda" else "cpu",
        devices=1,
        max_epochs=EPOCHS,
        precision="bf16-mixed" if DEVICE.type == "cuda" else 32,
        gradient_clip_val=1.0,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[ckpt_cb, EarlyStopping(monitor="val_loss", patience=3, mode="min")]
    )

    trainer.fit(model, dl, vl)

    best_path = ckpt_cb.best_model_path
    if best_path and os.path.exists(best_path):
        model = AblationAE.load_from_checkpoint(best_path, map_location=DEVICE)
        try:
            os.remove(best_path)
        except Exception:
            pass

    model = model.to(DEVICE).eval()
    return model


def evaluate_model(model, pred, ev, src, tgt):
    datasets = {0: [], 1: [], 2: []}
    z_all, z_f = [], []

    for fi in range(3):
        d = pred[fi]
        feats_all = d["feats"]
        ts_all = d["ts"]
        status_all = d["status"]
        ds_all = d["ds"]
        asset_all = d["asset"]

        ev_f = ev[FARMS[fi]]

        for j in np.unique(ds_all):
            m = ds_all == j

            feats = feats_all[m]
            ts = ts_all[m]
            status = status_all[m]

            if len(feats) < SEQ_LEN:
                continue

            w = windows(feats, 1)
            xt = torch.from_numpy(w).to(DEVICE)

            scores = []
            zz = []

            with torch.no_grad():
                for i in range(0, len(xt), 16384):
                    xb = xt[i:i + 16384].float()
                    xr, z = model(xb)

                    err = F.mse_loss(xr, xb, reduction="none").mean(dim=(1, 2))
                    scores.append(err.cpu().numpy().astype(np.float32))
                    zz.append(z.cpu().numpy().astype(np.float32))

            scores = np.concatenate(scores)
            zc = np.concatenate(zz)

            gt = (~np.isin(status, [0, 2])).astype(int)[SEQ_LEN - 1:]
            asset = str(asset_all[m][0]).strip()

            dd = dict(
                ts=ts[SEQ_LEN - 1:],
                scores=scores,
                gt=gt,
                asset=asset
            )

            e2 = ev_f[ev_f["_asset_key"] == asset]
            t0n = dd["ts"].min()
            t1n = dd["ts"].max()

            if len(e2) == 0:
                ov = np.array([], dtype=bool)
            else:
                ov = (e2["event_start"].values <= t1n) & (e2["event_end"].values >= t0n)

            anomaly_rows = e2[ov]
            anomaly_rows = anomaly_rows[anomaly_rows["event_label"].astype(str).str.strip() == "anomaly"]

            dd["is_anom"] = len(anomaly_rows) > 0
            dd["ev_start"] = anomaly_rows["event_start"].values[0] if dd["is_anom"] else None
            dd["ev_end"] = anomaly_rows["event_end"].values[0] if dd["is_anom"] else None

            datasets[fi].append(dd)

            z_all.append(zc[:500])
            z_f.append(np.full(min(len(zc), 500), fi))

    src_ds = []
    for k in src:
        src_ds += datasets[k]

    test_ds = datasets[tgt]

    normal_scores = np.concatenate([d["scores"][d["gt"] == 0] for d in src_ds])

    if len(normal_scores) == 0:
        normal_scores = np.concatenate([d["scores"] for d in src_ds])

    best = -1.0
    bthr = None

    for p in [95, 97, 99, 99.5, 99.9]:
        thr = float(np.percentile(normal_scores, p))
        r = care(src_ds, thr)
        selection_score = r["ef"] + r["fb"]

        if selection_score > best:
            best = selection_score
            bthr = thr

    r = care(test_ds, bthr)

    allsc = np.concatenate([d["scores"] for d in test_ds])
    allgt = np.concatenate([d["gt"] for d in test_ds])
    prc = (allsc > bthr).astype(int)

    try:
        auc = roc_auc_score(allgt, allsc)
    except Exception:
        auc = float("nan")

    prec = precision_score(allgt, prc, zero_division=0)
    rec = recall_score(allgt, prc, zero_division=0)

    Z = np.concatenate(z_all)
    FID = np.concatenate(z_f)

    if len(Z) > 20000:
        idx = np.random.default_rng(42).choice(len(Z), size=20000, replace=False)
        Z = Z[idx]
        FID = FID[idx]

    try:
        Xtr, Xte, ytr, yte = train_test_split(Z, FID, test_size=0.3, stratify=FID, random_state=42)
        dom = LogisticRegression(max_iter=1000).fit(Xtr, ytr).score(Xte, yte)
    except Exception:
        dom = float("nan")

    return dict(
        CARE=r["care"],
        AUC=auc,
        Recall=rec,
        Precision=prec,
        Fbeta=r["fb"],
        WS=r["ws"],
        EFbeta=r["ef"],
        Acc=r["acc"],
        DomainAcc=dom,
        threshold=bthr
    )


def config_name(phys, mmd, coral):
    name = "Physics AE" if phys else "Standard AE"

    if mmd and coral:
        name += " + MMD + CORAL"
    elif mmd:
        name += " + MMD"
    elif coral:
        name += " + CORAL"

    return name


def main():
    print("=" * 100)
    print("ABLATION STUDY")
    print("=" * 100)
    print(f"DEVICE: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"EPOCHS per run: {EPOCHS}")

    required = ["v3_train_X.npy", "v3_train_F.npy"]
    for f in FARMS:
        required.append(f"v3_pred_{f.replace(' ', '_')}.npy")

    missing = [f for f in required if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(f"Missing required files: {missing}")

    results = load_results()

    trX_phys = np.load("v3_train_X.npy")
    trF = np.load("v3_train_F.npy")

    pred_phys = load_pred_phys()
    ev = load_event_info()

    total = len(CONFIGS) * len(DIRECTIONS) * len(SEEDS)
    done = len(results)

    print(f"Total runs: {total}, already done: {done}")

    for dname, src, tgt in DIRECTIONS:
        print("\n" + "=" * 100)
        print(f"Building direction: {dname}")
        print("=" * 100)

        dd = build_direction(src, tgt, trX_phys, trF, pred_phys)

        phys_train_t = torch.from_numpy(dd["phys_train"]).to(DEVICE)
        std_train_t = torch.from_numpy(dd["std_train"]).to(DEVICE)
        train_fid_t = torch.from_numpy(dd["train_fid"]).to(DEVICE)

        for phys, mmd, coral in CONFIGS:
            cname = config_name(phys, mmd, coral)

            train_X_t = phys_train_t if phys else std_train_t
            pred_use = dd["pred_phys"] if phys else dd["pred_std"]

            for seed in SEEDS:
                if len(results) > 0:
                    exists = (
                        (results["config"] == cname) &
                        (results["direction"] == dname) &
                        (results["seed"] == seed)
                    ).any()

                    if exists:
                        continue

                run_t0 = time.time()

                model = train_model(
                    train_X_t,
                    train_fid_t,
                    use_mmd=mmd,
                    use_coral=coral,
                    seed=seed
                )

                metrics = evaluate_model(model, pred_use, ev, src, tgt)

                row = dict(
                    config=cname,
                    phys=int(phys),
                    mmd=int(mmd),
                    coral=int(coral),
                    direction=dname,
                    seed=seed,
                    CARE=metrics["CARE"],
                    AUC=metrics["AUC"],
                    Recall=metrics["Recall"],
                    Precision=metrics["Precision"],
                    Fbeta=metrics["Fbeta"],
                    WS=metrics["WS"],
                    EFbeta=metrics["EFbeta"],
                    Acc=metrics["Acc"],
                    DomainAcc=metrics["DomainAcc"],
                    threshold=metrics["threshold"],
                    time_s=time.time() - run_t0
                )

                results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
                results.to_csv(RESULTS_CSV, index=False)

                done += 1

                print(
                    f"[{done}/{total}] {cname:26} | {dname} | seed={seed} | "
                    f"CARE={metrics['CARE']:.4f} AUC={metrics['AUC']:.4f} "
                    f"Rec={metrics['Recall']:.3f} Dom={metrics['DomainAcc']:.3f} "
                    f"({metrics['time_s']:.0f}s)"
                )

                del model
                if DEVICE.type == "cuda":
                    torch.cuda.empty_cache()

        del phys_train_t, std_train_t, train_fid_t, dd
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("FINAL ABLATION SUMMARY (mean across seeds)")
    print("=" * 100)

    if len(results) > 0:
        summary = results.groupby(["config", "direction"]).agg(
            CARE=("CARE", "mean"),
            AUC=("AUC", "mean"),
            Recall=("Recall", "mean"),
            DomainAcc=("DomainAcc", "mean")
        ).round(4)

        print(summary.to_string())
    else:
        print("No results found.")

    print(f"\nResults saved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()