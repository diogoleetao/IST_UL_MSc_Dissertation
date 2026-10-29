import os, json, time, csv, argparse
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from models_thesis import build_model, Encoder
LR0 = 0.0001
BATCH = 12
LR_FACTOR = 0.7
N_EPOCHS = 25
PATIENCE = 8
MIN_LR = 1e-07
CLOUD_THRESH = 30.0
PATCH_CROP = 384
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_num_threads(6)
OUT = Path(__file__).parent
PIXCACHE_DIR = Path(os.environ.get('TESE_CACHE', 'C:/Users/diogo/tese_cache'))
PIXCACHE_NPY = PIXCACHE_DIR / 'splits_4band_384.npy'
PIXCACHE_IDX = PIXCACHE_DIR / 'splits_4band_384_index.json'

def find_dataset():
    for d in 'EFGDHC':
        cand = Path(f'{d}:/Tese/38_Cloud/38-Cloud_training')
        if (cand / 'train_red').is_dir():
            return cand
    raise FileNotFoundError('38-Cloud dataset not found on E:/F:/G:/D:/H:/C:')

def load_crop(base, stem):

    def rd(band):
        return cv2.imread(str(base / f'train_{band}' / f'{band}_{stem}'), cv2.IMREAD_UNCHANGED)
    img = np.stack([rd('red'), rd('green'), rd('blue'), rd('nir')], axis=-1)
    h, w = img.shape[:2]
    sh, sw = ((h - PATCH_CROP) // 2, (w - PATCH_CROP) // 2)
    return img[sh:sh + PATCH_CROP, sw:sw + PATCH_CROP]

def load_gt_frac(base, stem):
    gt = cv2.imread(str(base / 'train_gt' / f'gt_{stem}'), cv2.IMREAD_UNCHANGED)
    h, w = gt.shape[:2]
    sh, sw = ((h - PATCH_CROP) // 2, (w - PATCH_CROP) // 2)
    gt = gt[sh:sh + PATCH_CROP, sw:sw + PATCH_CROP]
    return (gt, 100.0 * float((gt > 127).mean()))

def build_pixcache(stems, base):
    PIXCACHE_DIR.mkdir(parents=True, exist_ok=True)
    stems = sorted(set(stems))
    if PIXCACHE_NPY.exists() and PIXCACHE_IDX.exists():
        have = json.load(open(PIXCACHE_IDX))
        if set(have['index'].keys()) >= set(stems):
            print(f"pixcache hit ({len(have['index'])} patches)")
            return (have['index'], have['cloud'])
    print(f'Building pixcache for {len(stems)} patches (one-time read)...', flush=True)
    arr = np.lib.format.open_memmap(PIXCACHE_NPY, mode='w+', dtype=np.uint16, shape=(len(stems), PATCH_CROP, PATCH_CROP, 4))
    index, cloud = ({}, {})
    t0 = time.time()
    for row, stem in enumerate(stems):
        arr[row] = load_crop(base, stem)
        _, cloud[stem] = load_gt_frac(base, stem)
        index[stem] = row
        if (row + 1) % 250 == 0:
            print(f'  {row + 1}/{len(stems)} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    arr.flush()
    del arr
    json.dump({'index': index, 'cloud': cloud}, open(PIXCACHE_IDX, 'w'))
    print(f'pixcache built in {(time.time() - t0) / 60:.1f} min')
    return (index, cloud)

def prep_arrays(stems, index, cloud, bands, size, base, seg=False):
    mm = np.load(PIXCACHE_NPY, mmap_mode='r')
    imgs = np.empty((len(stems), size, size, bands), dtype=np.float32)
    labels = np.array([[1.0 if cloud[s] >= CLOUD_THRESH else 0.0] for s in stems], np.float32)
    masks = np.empty((len(stems), size, size, 1), dtype=np.float32) if seg else None
    for i, stem in enumerate(stems):
        full = np.asarray(mm[index[stem]])[:, :, :bands]
        r = cv2.resize(full, (size, size), interpolation=cv2.INTER_AREA)
        imgs[i] = r.astype(np.float32) / 255.0
        if seg:
            gt, _ = load_gt_frac(base, stem)
            gtr = cv2.resize(gt.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)
            masks[i, :, :, 0] = (gtr > 127).astype(np.float32)
    del mm
    return (imgs, labels, masks)

class DS(Dataset):

    def __init__(self, imgs, labels, masks=None, augment=False):
        self.imgs, self.labels, self.masks, self.aug = (imgs, labels, masks, augment)
        self.seg = masks is not None

    def __len__(self):
        return len(self.imgs)

    def _aug(self, arr, fh, fv, k):
        if fh:
            arr = arr[:, ::-1]
        if fv:
            arr = arr[::-1, :]
        if k:
            arr = np.rot90(arr, k)
        return np.ascontiguousarray(arr)

    def __getitem__(self, i):
        img = self.imgs[i]
        fh = self.aug and np.random.rand() < 0.5
        fv = self.aug and np.random.rand() < 0.5
        k = np.random.randint(4) if self.aug else 0
        img = self._aug(img, fh, fv, k)
        x = torch.from_numpy(img).permute(2, 0, 1)
        if self.seg:
            m = self._aug(self.masks[i], fh, fv, k)
            return (x, torch.from_numpy(m).permute(2, 0, 1))
        return (x, torch.from_numpy(self.labels[i]))

def jaccard_loss(logits, target, smooth=1.0):
    p = torch.sigmoid(logits)
    inter = (p * target).sum()
    total = p.sum() + target.sum()
    jac = (inter + smooth) / (total - inter + smooth)
    return 1.0 - jac

def eval_clf(model, loader, crit):
    model.eval()
    P, Y, ls = ([], [], 0.0)
    with torch.no_grad():
        for x, y in loader:
            x, y = (x.to(DEVICE), y.to(DEVICE))
            out = model(x)
            ls += crit(out, y).item()
            P.append(torch.sigmoid(out).cpu().numpy().ravel())
            Y.append(y.cpu().numpy().ravel())
    p = np.concatenate(P)
    y = np.concatenate(Y).astype(int)
    pred = (p >= 0.5).astype(int)
    return dict(loss=ls / len(loader), acc=accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0), prec=precision_score(y, pred, zero_division=0), rec=recall_score(y, pred, zero_division=0), iou=float('nan'), p=p, y=y)

def eval_seg(model, loader, crit, decision_labels):
    model.eval()
    preds, ious, ls = ([], [], 0.0)
    thr = CLOUD_THRESH / 100.0
    with torch.no_grad():
        for x, m in loader:
            x, m = (x.to(DEVICE), m.to(DEVICE))
            out = model(x)
            ls += crit(out, m).item()
            pb = torch.sigmoid(out) > 0.5
            gt = m > 0.5
            for b in range(x.shape[0]):
                pp, gg = (pb[b, 0], gt[b, 0])
                inter = (pp & gg).sum().item()
                union = (pp | gg).sum().item()
                ious.append(inter / union if union > 0 else 1.0)
                preds.append(1 if pp.float().mean().item() >= thr else 0)
    y = decision_labels.ravel().astype(int)
    pred = np.array(preds)
    return dict(loss=ls / len(loader), acc=accuracy_score(y, pred), f1=f1_score(y, pred, zero_division=0), prec=precision_score(y, pred, zero_division=0), rec=recall_score(y, pred, zero_division=0), iou=float(np.mean(ious)), p=pred.astype(float), y=y)

def read_split(splits_dir, name):
    return [l.strip() for l in open(Path(splits_dir) / f'split_{name}.txt') if l.strip()]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='encoder', choices=['encoder', 'fcn', 'faithful', 'encoder_faithful'])
    ap.add_argument('--input', type=int, required=True)
    ap.add_argument('--bands', type=int, required=True, choices=[3, 4])
    ap.add_argument('--width', type=float, default=1.0)
    ap.add_argument('--train_size', type=int, default=2000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--splits_dir', default=str(OUT))
    ap.add_argument('--workers', type=int, default=0, help='0 is fastest: data is preloaded to RAM, so DataLoader workers only add spawn/pickle overhead (esp. on Windows)')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--widths', default=None, help='comma-separated per-stage channel widths (encoder only); overrides --width, e.g. 10,20,40,70,130 for DPU channel alignment')
    a = ap.parse_args()
    seg = a.model in ('fcn', 'faithful')
    tag = a.tag or f"{a.model}_{a.input}_{a.bands}b_w{str(a.width).replace('0.', '').replace('.', '')}"
    try:
        base = find_dataset()
    except FileNotFoundError:
        base = None
    train = read_split(a.splits_dir, 'train')[:a.train_size]
    val = read_split(a.splits_dir, 'calib')
    test = read_split(a.splits_dir, 'test')
    print(f'=== {tag} ===\ntrain={len(train)} val(calib)={len(val)} test={len(test)}')
    index, cloud = build_pixcache(train + val + test, base)
    print('pre-resizing train/val/test to RAM...', flush=True)
    tr_x, tr_y, tr_m = prep_arrays(train, index, cloud, a.bands, a.input, base, seg)
    va_x, va_y, va_m = prep_arrays(val, index, cloud, a.bands, a.input, base, seg)
    te_x, te_y, te_m = prep_arrays(test, index, cloud, a.bands, a.input, base, seg)
    tl = DataLoader(DS(tr_x, tr_y, tr_m, augment=True), batch_size=BATCH, shuffle=True, num_workers=a.workers)
    vl = DataLoader(DS(va_x, va_y, va_m, augment=False), batch_size=BATCH, shuffle=False, num_workers=a.workers)
    tel = DataLoader(DS(te_x, te_y, te_m, augment=False), batch_size=BATCH, shuffle=False, num_workers=a.workers)
    ev = (lambda mdl, ldr, lbl: eval_seg(mdl, ldr, crit, lbl)) if seg else lambda mdl, ldr, lbl: eval_clf(mdl, ldr, crit)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)
    if a.widths:
        wl = [int(x) for x in a.widths.split(',')]
        model = Encoder(a.bands, widths=wl).to(DEVICE)
    else:
        model = build_model(a.model, a.bands, a.width).to(DEVICE)
    n_params = sum((p.numel() for p in model.parameters()))
    crit = jaccard_loss if seg else nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=LR_FACTOR, patience=PATIENCE, min_lr=MIN_LR)
    print(f"model={a.model} input={a.input} bands={a.bands} width={a.width} params={n_params / 1000000.0:.4f}M  (loss={('jaccard' if seg else 'BCE')})")
    best_f1, best_state = (0.0, None)
    t0 = time.time()
    for ep in range(N_EPOCHS):
        model.train()
        ls = 0.0
        for x, y in tl:
            x, y = (x.to(DEVICE), y.to(DEVICE))
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            ls += loss.item()
        v = ev(model, vl, va_y)
        sched.step(v['loss'])
        mk = ''
        if v['f1'] > best_f1:
            best_f1, best_state = (v['f1'], {k: t.clone() for k, t in model.state_dict().items()})
            mk = '  +best'
        lr_now = opt.param_groups[0]['lr']
        iou_s = '' if np.isnan(v['iou']) else f" iou={v['iou']:.4f}"
        print(f"ep {ep + 1:2d} tloss={ls / len(tl):.4f} vloss={v['loss']:.4f} acc={v['acc']:.4f} f1={v['f1']:.4f}{iou_s} lr={lr_now:.1e}{mk}", flush=True)
    model.load_state_dict(best_state)
    save = OUT / f'{tag}.pth'
    torch.save(best_state, save)
    te = ev(model, tel, te_y)
    mins = (time.time() - t0) / 60
    iou_s = '' if np.isnan(te['iou']) else f" iou={te['iou']:.4f}"
    print(f"\nTEST (clean 500): acc={te['acc']:.4f} f1={te['f1']:.4f}{iou_s}  ({mins:.1f} min) -> {save.name}")
    header = ['tag', 'model', 'input', 'bands', 'width', 'train_size', 'seed', 'params_M', 'test_acc', 'test_f1', 'test_iou', 'test_prec', 'test_rec', 'minutes']
    row = [tag, a.model, a.input, a.bands, a.width, a.train_size, a.seed, round(n_params / 1000000.0, 4), round(te['acc'], 4), round(te['f1'], 4), '' if np.isnan(te['iou']) else round(te['iou'], 4), round(te['prec'], 4), round(te['rec'], 4), round(mins, 1)]
    csvp = OUT / 'train_results.csv'
    new = not csvp.exists()
    with open(csvp, 'a', newline='') as f:
        w = csv.writer(f)
        if new:
            w.writerow(header)
        w.writerow(row)
    print(f'Appended -> {csvp.name}')
if __name__ == '__main__':
    main()
