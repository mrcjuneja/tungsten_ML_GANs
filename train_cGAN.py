#!/usr/bin/env python3
import os, io, csv, math, json, time, shutil, zipfile, random
from pathlib import Path
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as SN
from torch.utils.data import Dataset, DataLoader

try:
    from cleanfid import fid as cleanfid
    HAVE_CLEANFID = True
except Exception:
    HAVE_CLEANFID = False

from skimage.metrics import structural_similarity as ssim

import copy

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model):
        d = self.decay
        for p_ema, p in zip(self.shadow.parameters(), model.parameters()):
            p_ema.data.mul_(d).add_(p.data, alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, model):
        for p, p_ema in zip(model.parameters(), self.shadow.parameters()):
            p.data.copy_(p_ema.data)

# ---------------------------
# Utils
# ---------------------------
def seed_everything(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def to_uint8(x):  # x in [-1,1] torch (B,1,H,W)
    x = x.clamp(-1,1).add(1).mul(127.5).add(0.5).to(torch.uint8)
    return x  # (B,1,H,W) uint8

def make_rgb_from_gray_uint8(g):  # g: (H,W) uint8 -> (H,W,3) uint8
    g = np.asarray(g, dtype=np.uint8)
    return np.stack([g,g,g], axis=2)

# ---------------------------
# Dataset: Dir or Zip + CSV labels (hf,temp)
# ---------------------------
class PhysCondDataset(Dataset):
    def __init__(self, data_path, img_size=256, grayscale=True,
                 labels_csv=None, cond_mode='csv', max_items=None):

        self.data_path = str(data_path)
        self.img_size = int(img_size)
        self.grayscale = grayscale
        self.cond_mode = cond_mode
        self._is_zip = self.data_path.lower().endswith(".zip")

        
        if self._is_zip:
            with zipfile.ZipFile(self.data_path, 'r') as zf:
                members = [m for m in zf.namelist()
                           if m.lower().endswith(('.png','.jpg','.jpeg','.tif','.tiff'))]
            self.members = sorted(members)
        else:
            p = Path(self.data_path)
            imgs = [str(x.relative_to(p)) for x in p.rglob("*")
                    if x.suffix.lower() in ('.png','.jpg','.jpeg','.tif','.tiff')]
            self.members = sorted(imgs)

        if max_items is not None:
            self.members = self.members[:max_items]

        # Labels
        self.cond_dim = 0
        self.disc_classes = None
        self.disc_to_idx = None
        self.cond_mean = None
        self.cond_std = None
        self.labels = None

        if self.cond_mode == 'csv':
            if labels_csv is None:
                raise ValueError("--labels_csv is required for cond_mode='csv'")
            self.labels = self._load_csv(os.path.abspath(labels_csv))
            # build cond matrix aligned to members
            conds = []
            for m in self.members:
                key = m.replace("\\","/")
                if key not in self.labels:
                    # also try without leading folders (basename match)
                    base = os.path.basename(key)
                    if base in self.labels:
                        key = base
                    else:
                        raise KeyError(f"CSV missing label for: {m}")
                hf, temp = self.labels[key]
                conds.append([float(hf), float(temp)])
            self.conds = np.array(conds, dtype=np.float32)
            self.cond_mean = self.conds.mean(axis=0)
            self.cond_std  = self.conds.std(axis=0) + 1e-8
            self.cond_dim = 2

        elif self.cond_mode == 'discrete':
            # infer class from first dir component
            classes = []
            for m in self.members:
                cls = m.split('/')[0]
                classes.append(cls)
            uniq = sorted(set(classes))
            self.disc_classes = uniq
            self.disc_to_idx = {c:i for i,c in enumerate(uniq)}
            self.cls_idx = np.array([self.disc_to_idx[m.split('/')[0]] for m in self.members], dtype=np.int64)
            self.cond_dim = len(uniq)

        else:
            self.cond_dim = 0  # unconditional

        # keep an open handle for zip (lazy)
        self._zip = None

    def _load_csv(self, csv_path):
        tab = {}
        with open(csv_path, 'r', newline='') as f:
            rdr = csv.DictReader(f)
            need = {'path','hf','temp'}
            if not need.issubset(set(rdr.fieldnames or [])):
                raise ValueError("CSV must have columns: path,hf,temp")
            for row in rdr:
                key = row['path'].strip().replace("\\","/")
                tab[key] = (float(row['hf']), float(row['temp']))
        return tab

    def __len__(self): return len(self.members)

    def _open_zip(self):
        if (self._zip is None) and self._is_zip:
            self._zip = zipfile.ZipFile(self.data_path, 'r')

    def _read_image(self, relpath):
        if self._is_zip:
            self._open_zip()
            with self._zip.open(relpath, 'r') as f:
                im = Image.open(io.BytesIO(f.read()))
        else:
            im = Image.open(os.path.join(self.data_path, relpath))
        im = im.convert('L') if self.grayscale else im.convert('RGB')
        im = im.resize((self.img_size, self.img_size), resample=Image.BILINEAR)
        return im

    def __getitem__(self, idx):
        m = self.members[idx]
        im = self._read_image(m)
        arr = np.asarray(im, dtype=np.uint8)
        if self.grayscale:
            arr = arr[None, ...]  # (1,H,W)
        else:
            arr = arr.transpose(2,0,1)  # (C,H,W)
        # [-1,1] float32
        x = torch.from_numpy(arr).float() / 127.5 - 1.0

        if self.cond_mode == 'csv':
            c = (self.conds[idx] - self.cond_mean) / self.cond_std
            c = torch.from_numpy(c.astype(np.float32))
            return x, c
        elif self.cond_mode == 'discrete':
            y = torch.tensor(self.cls_idx[idx], dtype=torch.long)
            return x, y
        else:
            return x, None

# ---------------------------
# Models
# ---------------------------
class GenCSV(nn.Module):
    def __init__(self, nz=128, out_ch=1, cond_dim=2):
        super().__init__()
        self.nz = nz
        self.cond_dim = cond_dim
        self.fc_cond = nn.Linear(cond_dim, nz) if cond_dim>0 else None
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz, 512, 4,1,0), nn.BatchNorm2d(512), nn.ReLU(True),  # 1→4
            nn.ConvTranspose2d(512,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(True),   # 4→8
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(True),   # 8→16
            nn.ConvTranspose2d(128, 64,4,2,1), nn.BatchNorm2d(64),  nn.ReLU(True),   # 16→32
            nn.ConvTranspose2d(64,  32,4,2,1), nn.BatchNorm2d(32),  nn.ReLU(True),   # 32→64
            nn.ConvTranspose2d(32, out_ch,4,2,1), nn.Tanh()                            # 64→128; add one more up to 256 below
        )
        self.upsample_to_256 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)  # 128→256
        )

    def forward(self, z, cond=None):
        if (self.fc_cond is not None) and (cond is not None):
            z = z + self.fc_cond(cond).unsqueeze(2).unsqueeze(3)  # (B,nz,1,1)
        x = self.net(z)
        x = self.upsample_to_256(x)
        return x

class GenDisc(nn.Module):
    def __init__(self, nz=128, out_ch=1, num_classes=8):
        super().__init__()
        self.emb = nn.Embedding(num_classes, nz)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(nz,512,4,1,0), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.ConvTranspose2d(512,256,4,2,1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256,128,4,2,1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64,4,2,1), nn.BatchNorm2d(64),  nn.ReLU(True),
            nn.ConvTranspose2d(64,  32,4,2,1), nn.BatchNorm2d(32),  nn.ReLU(True),
            nn.ConvTranspose2d(32, out_ch,4,2,1), nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )
    def forward(self, z, labels):
        z = z + self.emb(labels).view(z.size(0), z.size(1), 1,1)
        return self.net(z)

class DiscCSV(nn.Module):
    def __init__(self, in_ch=1, cond_dim=2):
        super().__init__()
        self.cond_dim = cond_dim
        self.feat = nn.Sequential(
            SN(nn.Conv2d(in_ch,64,4,2,1)),  nn.LeakyReLU(0.2,True),   # 256→128
            SN(nn.Conv2d(64,128,4,2,1)),   nn.LeakyReLU(0.2,True),   # 128→64
            SN(nn.Conv2d(128,256,4,2,1)),  nn.LeakyReLU(0.2,True),   # 64→32
            SN(nn.Conv2d(256,256,4,2,1)),  nn.LeakyReLU(0.2,True),   # 32→16
            SN(nn.Conv2d(256,256,4,2,1)),  nn.LeakyReLU(0.2,True),   # 16→8
            SN(nn.Conv2d(256,256,4,2,1)),  nn.LeakyReLU(0.2,True),   # 8→4
        )
        self.out = SN(nn.Conv2d(256,1,4,1,0))  # → scalar per-sample
        self.proj = nn.Linear(cond_dim, 256) if cond_dim>0 else None

    def forward(self, x, cond=None):
        f = self.feat(x)                 # (B,256,4,4)
        u = self.out(f).view(x.size(0))  # (B,)
        if (self.proj is not None) and (cond is not None):
            v = f.view(f.size(0), 256, -1).sum(2)   # (B,256) global pooled
            p = (v * self.proj(cond)).sum(1)        # (B,)
            return u + p
        return u

class PatchD(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()
        self.net = nn.Sequential(
            SN(nn.Conv2d(in_ch,64,4,2,1)),  nn.LeakyReLU(0.2,True),   # 256→128
            SN(nn.Conv2d(64,128,4,2,1)),   nn.LeakyReLU(0.2,True),   # 128→64
            SN(nn.Conv2d(128,256,4,2,1)),  nn.LeakyReLU(0.2,True),   # 64→32
            SN(nn.Conv2d(256,1,4,1,1))                                  # 32→32 map
        )
    def forward(self, x): return self.net(x)

# ---------------------------
# Losses
# ---------------------------
def hinge_d_loss(real_scores, fake_scores):
    return F.relu(1.0 - real_scores).mean() + F.relu(1.0 + fake_scores).mean()

def hinge_g_loss(fake_scores):
    return -fake_scores.mean()

# ---------------------------
# Metrics
# ---------------------------
import tempfile
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
import torchvision.utils as vutils

def to_uint8_01(x):
    if isinstance(x, torch.Tensor):
        x = x.detach()
        if x.dtype.is_floating_point:
            if x.min() < -0.01 or x.max() > 1.01:   # likely [-1,1]
                x = (x.clamp(-1,1) + 1) / 2
            x = (x * 255.0 + 0.5).clamp(0,255).to(torch.uint8)
        elif x.dtype != torch.uint8:
            x = x.to(torch.uint8)
    return x

def ensure_3ch_uint8(x_u8):
    if x_u8.ndim == 3:
        x_u8 = x_u8.unsqueeze(1)
    if x_u8.shape[1] == 1:
        x_u8 = x_u8.repeat(1,3,1,1)
    return x_u8

def write_png_dir(tensor_u8_BCHW, out_dir, prefix):
    os.makedirs(out_dir, exist_ok=True)
    B = tensor_u8_BCHW.shape[0]
    for i in range(B):
        arr = tensor_u8_BCHW[i].permute(1,2,0).cpu().numpy()  # HWC uint8
        Image.fromarray(arr).save(os.path.join(out_dir, f"{prefix}{i:06d}.png"))

def compute_ssim_batch_robust(real_u8_B1HW, fake_u8_B1HW):
    B = min(real_u8_B1HW.shape[0], fake_u8_B1HW.shape[0])
    if B == 0: return float('nan')
    vals = []
    for i in range(B):
        r = real_u8_B1HW[i,0].cpu().numpy()
        f = fake_u8_B1HW[i,0].cpu().numpy()
        if r.shape != f.shape:
            H = min(r.shape[0], f.shape[0]); W = min(r.shape[1], f.shape[1])
            r = r[:H,:W]; f = f[:H,:W]
        try:
            v = ssim(r, f, data_range=255,
                     gaussian_weights=True, use_sample_covariance=False)
            if not np.isnan(v):
                vals.append(v)
        except Exception:
            pass
    return float(np.mean(vals)) if vals else float('nan')

def try_cleanfid(real_dir, fake_dir, kid=True):
    if not HAVE_CLEANFID:
        return float('nan'), float('nan')
    try:
        if (not os.path.isdir(real_dir)) or (not os.listdir(real_dir)):  # empty
            return float('nan'), float('nan')
        if (not os.path.isdir(fake_dir)) or (not os.listdir(fake_dir)):
            return float('nan'), float('nan')
        fid_v = cleanfid.compute_fid(fake_dir, real_dir, mode="clean")
        kid_v = cleanfid.compute_kid(fake_dir, real_dir, mode="clean") if kid else float('nan')
        return float(fid_v), float(kid_v)
    except Exception:
        return float('nan'), float('nan')

def quick_grain_porosity_uint8(gray_u8_list):
    areas, poros = [], []
    for g in gray_u8_list:
        try:
            thr = threshold_otsu(g)
        except Exception:
            continue
        bw = (g > thr)
        lab = label(bw, connectivity=2)
        props = regionprops(lab)
        areas.extend([p.area for p in props if p.area > 5])
        poro = 1.0 - (bw.sum() / float(bw.size))
        poros.append(poro)
    med_area = float(np.median(areas)) if areas else float('nan')
    mean_poro = float(np.mean(poros)) if poros else float('nan')
    return med_area, mean_poro

# ---------------------------
# Training
# ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True, help='dataset dir or .zip')
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--labels_csv', default=None, help='CSV with path,hf,temp for cond')
    ap.add_argument('--cond', default='csv', choices=['csv','none','discrete'])
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--nz', type=int, default=128)
    ap.add_argument('--lr', type=float, default=2e-4)
    ap.add_argument('--seed', type=int, default=42)

    # eval cadence (epochs)
    ap.add_argument('--fid_every', type=int, default=2, help='compute FID/KID every N epochs')
    ap.add_argument('--ssim_every', type=int, default=1, help='compute SSIM/physics every N epochs')

    # snapshots cadence (epochs)
    ap.add_argument('--snap_every', type=int, default=2)

    ap.add_argument('--img_size', type=int, default=256)
    args = ap.parse_args()

    seed_everything(args.seed)
    ensure_dir(args.outdir)

    # dataset & loader
    ds = PhysCondDataset(args.data, img_size=args.img_size, grayscale=True,
                         labels_csv=args.labels_csv, cond_mode=args.cond)
    dl = DataLoader(ds, batch_size=args.batch, shuffle=True,
                    num_workers=args.workers, pin_memory=True, drop_last=True)
    print(f"Loaded {len(ds)} images from {args.data} | cond_mode={args.cond}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    nz = args.nz

    # models (unchanged from your version)
    if args.cond == 'csv':
        G = GenCSV(nz=nz, out_ch=1, cond_dim=2).to(device)
        ema = EMA(G, decay=0.999)
        D = DiscCSV(in_ch=1, cond_dim=2).to(device)
    elif args.cond == 'discrete':
        ncls = ds.cond_dim
        G = GenDisc(nz=nz, out_ch=1, num_classes=ncls).to(device)
        D = DiscCSV(in_ch=1, cond_dim=0).to(device)  # using global head without projection term
    else:
        G = GenCSV(nz=nz, out_ch=1, cond_dim=0).to(device)
        D = DiscCSV(in_ch=1, cond_dim=0).to(device)

    P = PatchD(in_ch=1).to(device)

    optG = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.0, 0.9))
    optD = torch.optim.Adam(list(D.parameters()) + list(P.parameters()), lr=args.lr, betas=(0.0, 0.9))

    # fixed grid noise/conds for snapshots
    grid_z = torch.randn(64, nz, 1, 1, device=device)
    if args.cond == 'csv':
        sel = np.linspace(0, len(ds)-1, 64).round().astype(int)
        conds = []
        for i in sel:
            c = (ds.conds[i] - ds.cond_mean) / ds.cond_std
            conds.append(c)
        grid_c = torch.from_numpy(np.array(conds, dtype=np.float32)).to(device)
    elif args.cond == 'discrete':
        sel = np.arange(64) % ds.cond_dim
        grid_c = torch.from_numpy(sel.astype(np.int64)).to(device)
    else:
        grid_c = None

    # logging
    log_path = os.path.join(args.outdir, "train_log.jsonl")
    with open(log_path, 'w'):
        pass

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        for real, cond in dl:
            real = real.to(device)  # (B,1,H,W) in [-1,1]
            if args.cond == 'csv':
                cond = cond.to(device)      # (B,2)
            elif args.cond == 'discrete':
                cond = cond.to(device)      # (B,)
            else:
                cond = None

            b = real.size(0)

            # ---------------- D step ----------------
            z = torch.randn(b, nz, 1, 1, device=device)
            with torch.no_grad():
                fake = G(z, cond)

            D.zero_grad(set_to_none=True)
            P.zero_grad(set_to_none=True)

            rf = D(real, cond)
            ff = D(fake.detach(), cond)
            d_glob = hinge_d_loss(rf, ff)

            dpr = P(real)
            dpf = P(fake.detach())
            d_patch = (F.relu(1.0 - dpr).mean() + F.relu(1.0 + dpf).mean())

            d_loss = d_glob + 0.5 * d_patch
            d_loss.backward()
            optD.step()

            # ---------------- G step ----------------
            z = torch.randn(b, nz, 1, 1, device=device)
            fake = G(z, cond)

            g_glob = hinge_g_loss(D(fake, cond))
            g_patch = -P(fake).mean()
            g_loss = g_glob + 0.5 * g_patch

            optG.zero_grad(set_to_none=True)
            g_loss.backward()
            optG.step()
            ema.update(G)


            # ---------------- Eval (SSIM/physics per N epochs) ----------------
            ssim_v = float('nan')
            fid_v  = float('nan')
            kid_v  = float('nan')
            med_area_r = med_area_f = float('nan')
            poro_r = poro_f = float('nan')

            step_in_epoch = (global_step % len(dl))

            if (epoch % max(1, args.ssim_every) == 0) and (step_in_epoch == 0):
                with torch.no_grad():
                    # Use a small eval batch
                    B_eval = min(b, 64)
                    real_eval = real[:B_eval]
                    cond_eval = (cond[:B_eval] if cond is not None else None)
                    z_eval = torch.randn(B_eval, nz, 1, 1, device=device)
                    G.eval()
                    fake_eval = G(z_eval, cond_eval)  # (B,1,H,W) in [-1,1]
                    G.train()

                    real_u8 = to_uint8_01(real_eval)     # (B,1,H,W) uint8
                    fake_u8 = to_uint8_01(fake_eval)     # (B,1,H,W) uint8

                    # SSIM on grayscale channel
                    ssim_v = compute_ssim_batch_robust(real_u8, fake_u8)

                    # quick physics snapshot on SAME batch
                    reals_gray = [real_u8[i, 0].cpu().numpy() for i in range(B_eval)]
                    fakes_gray = [fake_u8[i, 0].cpu().numpy() for i in range(B_eval)]
                    med_area_r, poro_r = quick_grain_porosity_uint8(reals_gray)
                    med_area_f, poro_f = quick_grain_porosity_uint8(fakes_gray)

            # ---------------- Eval (FID/KID per N epochs) ----------------
            if HAVE_CLEANFID and (epoch % max(1, args.fid_every) == 0) and (step_in_epoch == 0):
                with torch.no_grad():
                    B_eval = min(b, 64)
                    real_eval = real[:B_eval]
                    cond_eval = (cond[:B_eval] if cond is not None else None)
                    z_eval = torch.randn(B_eval, nz, 1, 1, device=device)
                    G.eval()
                    fake_eval = G(z_eval, cond_eval)
                    G.train()

                    # write both real & fake as RGB to temp dirs
                    real_u8 = to_uint8_01(real_eval)
                    fake_u8 = to_uint8_01(fake_eval)
                    real_rgb = ensure_3ch_uint8(real_u8)
                    fake_rgb = ensure_3ch_uint8(fake_u8)

                    with tempfile.TemporaryDirectory() as tmp:
                        real_dir = os.path.join(tmp, "real")
                        fake_dir = os.path.join(tmp, "fake")
                        write_png_dir(real_rgb, real_dir, "r")
                        write_png_dir(fake_rgb, fake_dir, "f")
                        fid_v, kid_v = try_cleanfid(real_dir, fake_dir, kid=True)

            # ---------------- Log ----------------
            rec = dict(
                step=int(global_step),
                epoch=int(epoch),
                d=float(d_loss.item()),
                g=float(g_loss.item()),
                ssim=float(ssim_v),
                fid=float(fid_v),
                kid=float(kid_v),
                med_grain_area_real=float(med_area_r),
                med_grain_area_fake=float(med_area_f),
                porosity_real=float(poro_r),
                porosity_fake=float(poro_f),
            )
            with open(log_path, 'a') as f:
                f.write(json.dumps(rec) + "\n")

            # ---------------- Snapshots (per N epochs) ----------------
            if (epoch % max(1, args.snap_every) == 0) and (step_in_epoch == 0):
                with torch.no_grad():
                    if args.cond == 'csv':
                        grid = G(grid_z, grid_c)
                    elif args.cond == 'discrete':
                        grid = G(grid_z, grid_c)
                    else:
                        grid = G(grid_z, None)
                    # save a nice 8x8 grid (grayscale)
                    vutils.save_image(
                        (grid.clamp(-1,1)+1)/2,
                        os.path.join(args.outdir, f"snap_e{epoch:03d}_s{global_step:07d}.png"),
                        nrow=8, normalize=False
                    )
                # ---- Save a matching checkpoint ----
                ckpt_path = os.path.join(args.outdir, f"ckpt_e{epoch:03d}_s{global_step:07d}.pt")
                state = {
                    "G": G.state_dict(),
                    "step": global_step,
                    "epoch": epoch,
                    "nz": args.nz,
                    "img_size": args.img_size,
                    "cond_mode": args.cond,  # 'csv' | 'discrete' | 'none'
                    "cond_dim": (2 if args.cond == 'csv' else (ds.cond_dim if args.cond == 'discrete' else 0)),
                    "norm_stats": {
                        "mean": (ds.cond_mean.tolist() if getattr(ds, 'cond_mean', None) is not None else None),
                        "std":  (ds.cond_std.tolist()  if getattr(ds, 'cond_std',  None) is not None else None),
                    },
                }
                
                if 'ema' in locals() and hasattr(ema, 'shadow'):
                    state["G_ema"] = ema.shadow.state_dict()
                torch.save(state, ckpt_path)
                print(f"Saved {ckpt_path}")

            global_step += 1

        print(f"Epoch {epoch}/{args.epochs} done in {time.time()-t0:.1f}s")

    print("Training finished. Logs:", log_path)

if __name__ == "__main__":
    main()

