import os, glob, json, time
import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_score, recall_score
import warnings; warnings.filterwarnings("ignore")

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True
pl.seed_everything(42)

BASE_DIR = "data/raw/CARE_To_Compare"
FARMS = ["Wind Farm A", "Wind Farm B", "Wind Farm C"]
SEQ_LEN = 36
BETA = 0.5
N_FEAT = 12
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature Indices (from v5 builder):
# 0:wind, 1:rotor, 2:power, 3:pitch, 4:ambient, 5:oil_rise, 6:brg_max_rise, 
# 7:brg_spread, 8:power_residual, 9:electrical, 10:hydraulic, 11:hyd_mask

class StreamEnc(nn.Module):
    def __init__(s, d, h=128, z=64):
        super().__init__()
        s.lstm = nn.LSTM(d, h, batch_first=True, bidirectional=True)
        s.fc = nn.Linear(h*2, z)
    def forward(s, x): return s.fc(s.lstm(x)[0][:, -1])

class DualStreamV5(pl.LightningModule):
    def __init__(s, max_epochs=20):
        super().__init__(); s.save_hyperparameters()
        s.max_epochs = max_epochs
        # Stream 1: Aero/Boundary (6 dims: wind, rotor, power, pitch, ambient, power_res)
        s.aero = StreamEnc(6)
        # Stream 2: Thermal/Aux (6 dims: oil, brg_max, brg_spread, elec, hyd, mask)
        s.therm = StreamEnc(6)
        
        s.fusion = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 128))
        s.dec = nn.Sequential(nn.Linear(128, 256), nn.ReLU(), nn.Linear(256, 512), 
                              nn.ReLU(), nn.Linear(512, SEQ_LEN * N_FEAT))
        
        # Physics forecasting parameters (log-space to keep positive)
        s.log_alpha_oil = nn.Parameter(torch.tensor(-4.5))
        s.log_beta_oil  = nn.Parameter(torch.tensor(-4.0))
        s.log_alpha_brg = nn.Parameter(torch.tensor(-4.5))
        s.log_beta_brg  = nn.Parameter(torch.tensor(-4.0))

    def forward(s, x):
        # Split streams
        x_aero = torch.cat([x[:,:,:5], x[:,:,8:9]], dim=-1) # 0,1,2,3,4 and 8
        x_therm = torch.cat([x[:,:,5:8], x[:,:,9:12]], dim=-1) # 5,6,7 and 9,10,11
        
        z = s.fusion(torch.cat([s.aero(x_aero), s.therm(x_therm)], -1))
        return s.dec(z).view(-1, SEQ_LEN, N_FEAT), z

    def _phys_params(s):
        return (torch.exp(s.log_alpha_oil), torch.exp(s.log_beta_oil),
                torch.exp(s.log_alpha_brg), torch.exp(s.log_beta_brg))

    def physics_forecast_loss(s, x):
        """Forecast next-step thermal rise. Apply ONLY to real data."""
        ao, bo, ab, bb = s._phys_params()
        P = x[:,:,2] # power_pu
        oil = x[:,:,5] # oil_rise
        brg = x[:,:,6] # brg_max_rise
        
        oil_hat = (1-bo)*oil[:,:-1] + ao*P[:,:-1]
        brg_hat = (1-bb)*brg[:,:-1] + ab*P[:,:-1]
        
        return F.mse_loss(oil_hat, oil[:,1:]) + F.mse_loss(brg_hat, brg[:,1:])

    def physics_residual(s, x):
        """Per-window physics forecast residual for anomaly scoring."""
        ao, bo, ab, bb = s._phys_params()
        P = x[:,:,2]; oil = x[:,:,5]; brg = x[:,:,6]
        r_oil = (oil[:,1:] - ((1-bo)*oil[:,:-1] + ao*P[:,:-1]))**2
        r_brg = (brg[:,1:] - ((1-bb)*brg[:,:-1] + ab*P[:,:-1]))**2
        return (r_oil.mean(1) + r_brg.mean(1))

    def mmd(s, x, y):
        def k(a,b):
            d = torch.cdist(a,b)**2
            return sum(torch.exp(-d/(2*s2)) for s2 in (1.0, 5.0, 10.0))
        return k(x,x).mean() + k(y,y).mean() - 2*k(x,y).mean()

    def training_step(s, b, i):
        x, fid = b
        xr, z = s(x)
        
        # 1. Weighted Reconstruction (Thermal/Aux weighted 3x)
        w = torch.tensor([1,1,1,1,1, 3,3,3, 1, 2,2,0], dtype=x.dtype, device=x.device)
        loss_recon = ((xr - x)**2 * w).mean()
        
        # 2. Physics Forecast Loss (on real data x, NOT reconstruction xr)
        loss_phys = s.physics_forecast_loss(x)
        
        # 3. MMD Domain Alignment
        p = float(i + s.current_epoch*s.trainer.num_training_batches)/(s.max_epochs*s.trainer.num_training_batches)
        lam = min(1.0, 2.0 * p) # Linear warmup
        
        da, npairs = 0.0, 0
        for a in range(3):
            for c in range(a+1, 3):
                za, zc = z[fid==a], z[fid==c]
                if len(za)>16 and len(zc)>16:
                    da += s.mmd(za, zc); npairs += 1
        
        loss = loss_recon + 1.0*loss_phys + (lam * (da/npairs) if npairs else 0.0)
        
        s.log('train_loss', loss, prog_bar=True)
        s.log('recon', loss_recon, prog_bar=True)
        s.log('phys', loss_phys, prog_bar=True)
        return loss

    def validation_step(s, b, i):
        x, fid = b
        xr, z = s(x)
        w = torch.tensor([1,1,1,1,1, 3,3,3, 1, 2,2,0], dtype=x.dtype, device=x.device)
        loss_recon = ((xr - x)**2 * w).mean()
        s.log('val_loss', loss_recon + s.physics_forecast_loss(x))

    def configure_optimizers(s):
        return torch.optim.AdamW(s.parameters(), lr=2e-3, weight_decay=1e-4)

def windows(feats, stride):
    T = len(feats)
    if T < SEQ_LEN: return np.empty((0, SEQ_LEN, feats.shape[1]), dtype=np.float32)
    w = np.lib.stride_tricks.sliding_window_view(feats, window_shape=SEQ_LEN, axis=0)
    return np.ascontiguousarray(w[::stride].transpose(0, 2, 1), dtype=np.float32)

def f_beta(p, r, beta=BETA):
    return (1+beta**2)*p*r/(beta**2*p+r) if (beta**2*p+r) else 0.0

def main():
    t0 = time.time()
    print("="*90)
    print("V5 TRAINING: 12-Feature Subsystem-Aligned Dual-Stream")
    print("="*90)
    print(f"DEVICE: {DEVICE}")
    
    trX = torch.from_numpy(np.load('v5_train_X.npy')).to(DEVICE)
    trF = torch.from_numpy(np.load('v5_train_F.npy')).to(DEVICE)
    print(f"Train tensor: {tuple(trX.shape)}")
    
    model = DualStreamV5(max_epochs=20)
    trainer = pl.Trainer(
        accelerator='gpu' if DEVICE.type=='cuda' else 'cpu', devices=1, max_epochs=20, 
        precision='bf16-mixed' if DEVICE.type=='cuda' else 32,
        gradient_clip_val=1.0, log_every_n_steps=20, num_sanity_val_steps=0,
        callbacks=[
            ModelCheckpoint('ckpt5', monitor='val_loss', mode='min'),
            EarlyStopping(monitor='val_loss', patience=6, mode='min')
        ]
    )
    train_dl = DataLoader(TensorDataset(trX, trF), batch_size=4096, shuffle=True, num_workers=0)
    val_dl   = DataLoader(TensorDataset(trX[:20000], trF[:20000]), batch_size=4096, num_workers=0)
    
    trainer.fit(model, train_dl, val_dl)
    best = DualStreamV5.load_from_checkpoint(trainer.checkpoint_callback.best_model_path).to(DEVICE).eval()
    
    # ---- Physics interpretability ----
    ao, bo, ab, bb = [t.item() for t in best._phys_params()]
    tau_oil = 10.0/bo; tau_brg = 10.0/bb
    print("\nLEARNED PHYSICS (v5)")
    print(f"Oil tau: {tau_oil:.0f} min | Bearing tau: {tau_brg:.0f} min")
    
    # ---- Zero-Shot Eval on Farm C ----
    ev = {}
    for f in FARMS:
        e = pd.read_csv(os.path.join(BASE_DIR, f, 'event_info.csv'), sep=None, engine='python')
        e.columns = e.columns.str.strip()
        e['event_start'] = pd.to_datetime(e['event_start']).astype('int64')
        e['event_end']   = pd.to_datetime(e['event_end']).astype('int64')
        ev[f] = e
        
    datasets = {f: [] for f in FARMS}; z_all, z_f = [], []
    for fi, farm in enumerate(FARMS):
        d = np.load(f'v5_pred_{farm.replace(" ", "_")}.npy', allow_pickle=True).item()
        for j in np.unique(d['ds']):
            m = d['ds']==j
            feats, ts, status = d['feats'][m], d['ts'][m], d['status'][m]
            if len(feats) < SEQ_LEN: continue
            w = windows(feats, 1)
            xt = torch.from_numpy(w).to(DEVICE)
            recon_s, phys_s, zz = [], [], []
            with torch.no_grad():
                for i in range(0, len(xt), 8192):
                    xb = xt[i:i+8192]
                    xr, z = best(xb)
                    # Thermal weighted recon error
                    wt = torch.tensor([1,1,1,1,1, 3,3,3, 1, 2,2,0], dtype=xb.dtype, device=xb.device)
                    err = ((xr - xb)**2 * wt).mean((1,2))
                    recon_s.append(err.cpu().numpy())
                    phys_s.append(best.physics_residual(xb).cpu().numpy())
                    zz.append(z.cpu().numpy())
            
            recon_s = np.concatenate(recon_s); phys_s = np.concatenate(phys_s)
            # Normalize and combine
            rs = (recon_s-recon_s.mean())/(recon_s.std()+1e-8)
            ps = (phys_s-phys_s.mean())/(phys_s.std()+1e-8)
            score = rs + 1.0*ps
            
            zc = np.concatenate(zz)
            datasets[farm].append(dict(ts=ts[SEQ_LEN-1:], scores=score,
                gt=(~np.isin(status,[0,2])).astype(int)[SEQ_LEN-1:], asset=d['asset'][m][0]))
            z_all.append(zc[:500]); z_f.append(np.full(min(len(zc),500), fi))
            
    for f in FARMS:
        edf = ev[f]; ac = 'asset' if 'asset' in edf.columns else 'asset_id'
        for d in datasets[f]:
            e2 = edf[edf[ac]==d['asset']]
            t0n, t1n = d['ts'].min(), d['ts'].max()
            ov = (e2['event_start'].values<=t1n)&(e2['event_end'].values>=t0n) if len(e2) else np.array([],bool)
            d['is_anom'] = bool(len(e2) and (e2[ov]['event_label']=='anomaly').any())
            d['ev_start'] = e2[ov]['event_start'].values[0] if d['is_anom'] else None
            d['ev_end']   = e2[ov]['event_end'].values[0]   if d['is_anom'] else None

    def care(ds_list, thr):
        fb, ws, acc, evp, evt, ear = [], [], [], [], [], []
        for d in ds_list:
            pr = (d['scores']>thr).astype(int); gt = d['gt']
            evp.append(int(pr.sum()>0)); evt.append(d['is_anom'])
            if d['is_anom']:
                tp=((pr==1)&(gt==1)).sum(); fp=((pr==1)&(gt==0)).sum(); fn=((pr==0)&(gt==1)).sum()
                fb.append(f_beta(tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0))
                if d['ev_start'] is not None:
                    mk=(d['ts']>=d['ev_start'])&(d['ts']<=d['ev_end'])
                    if mk.sum():
                        pos=(d['ts'][mk]-d['ev_start'])/max(d['ev_end']-d['ev_start'],1)
                        wts=np.where(pos<=0.5,1,np.clip(1-(pos-0.5)/0.5,0,1))
                        ws.append((wts*pr[mk]).sum()/wts.sum())
                    i0=np.argmax(pr==1) if pr.sum() else None
                    if i0 is not None and d['ts'][i0]<d['ev_start']:
                        ear.append((d['ev_start']-d['ts'][i0])/86400e9)
            else:
                acc.append((pr==gt).mean())
        fbm=np.mean(fb) if fb else 0; wsm=np.mean(ws) if ws else 0; acm=np.mean(acc) if acc else 0
        tp=((np.array(evp)==1)&(np.array(evt)==1)).sum(); fp=((np.array(evp)==1)&(np.array(evt)==0)).sum()
        fn=((np.array(evp)==0)&(np.array(evt)==1)).sum()
        ef=f_beta(tp/(tp+fp) if tp+fp else 0, tp/(tp+fn) if tp+fn else 0)
        carev = 0 if sum(evp)==0 else (acm if acm<0.5 else (fbm+wsm+ef+2*acm)/5)
        return dict(care=carev, fb=fbm, ws=wsm, ef=ef, acc=acm, ear=ear)

    # Calibrate on A+B
    cal = np.concatenate([d['scores'][d['gt']==0] for f in FARMS[:2] for d in datasets[f]])
    calds = datasets[FARMS[0]] + datasets[FARMS[1]]
    best_s, bthr = -1, None
    for p in [95,97,99,99.5,99.9]:
        r = care(calds, np.percentile(cal,p))
        if r['ef']+r['fb']>best_s: best_s, bthr = r['ef']+r['fb'], np.percentile(cal,p)
        
    # Test on C
    r = care(datasets[FARMS[2]], bthr)
    allsc = np.concatenate([d['scores'] for d in datasets[FARMS[2]]])
    allgt = np.concatenate([d['gt'] for d in datasets[FARMS[2]]])
    prc = (allsc>bthr).astype(int)
    
    Z, FID = np.concatenate(z_all), np.concatenate(z_f)
    Xtr,Xte,ytr,yte = train_test_split(Z,FID,test_size=0.3,stratify=FID,random_state=42)
    dom = LogisticRegression(max_iter=1000).fit(Xtr,ytr).score(Xte,yte)
    
    print("\n" + "="*90)
    print("FARM C ZERO-SHOT (v5: 12-Feature Subsystem + Forecasting + MMD)")
    print("="*90)
    print(f"CARE={r['care']:.4f} (fb={r['fb']:.3f} ws={r['ws']:.3f} ef={r['ef']:.3f} acc={r['acc']:.3f})")
    print(f"AUC={roc_auc_score(allgt,allsc):.4f} P={precision_score(allgt,prc,zero_division=0):.3f} R={recall_score(allgt,prc,zero_division=0):.3f}")
    if r['ear']: print(f"Earliness mean={np.mean(r['ear']):.2f}d max={np.max(r['ear']):.2f}d")
    print(f"Domain classifier acc={dom:.4f} (chance=0.3333)")
    print(f"TOTAL TIME: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()