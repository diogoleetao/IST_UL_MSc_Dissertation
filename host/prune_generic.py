import os, json, time, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
torch.nn.Module.cuda = lambda self, *a, **k: self
torch.Tensor.cuda = lambda self, *a, **k: self
from pytorch_nndct.pruning import IterativePruningRunner
import nndct_shared.pruning.sensitivity as _sensmod

def _load_sens_json(filepath):
    s = _sensmod.NetSensitivity()
    info = json.load(open(filepath, 'r'))
    s.groups = [_sensmod.GroupMetrics.deserialize(g) for g in info['groups']]
    s.uncompleted_steps = info['uncompleted_steps']
    s.graph_digest = info.get('graph_digest')
    return s
_sensmod.load_sens = _load_sens_json
INPUT, BANDS, PATCH_CROP = (96, 4, 384)
CLOUD_THRESH, PIX = (30.0, 127)
DEVICE = torch.device('cpu')

class Encoder4Band(nn.Module):

    def __init__(self, widths=(32, 64, 128, 256, 512), in_channels=4):
        super().__init__()
        w = list(widths)

        def blk(ic, oc, pool=True):
            L = [nn.Conv2d(ic, oc, 3, padding=1), nn.BatchNorm2d(oc), nn.ReLU(inplace=True)]
            if pool:
                L.append(nn.MaxPool2d(2))
            return nn.Sequential(*L)
        self.enc1 = blk(in_channels, w[0], pool=False)
        self.enc2 = blk(w[0], w[1])
        self.enc3 = blk(w[1], w[2])
        self.enc4 = blk(w[2], w[3])
        self.enc5 = blk(w[3], w[4])
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(w[4], 1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)
        x = self.enc4(x)
        x = self.enc5(x)
        return self.fc(self.gap(x).flatten(1))

def widths_of(state):
    return [int(state[f'enc{i}.0.weight'].shape[0]) for i in range(1, 6)]

def preload(cache_npy, cache_json, stems):
    meta = json.load(open(cache_json))
    index, cloud = (meta['index'], meta['cloud'])
    mm = np.load(cache_npy, mmap_mode='r')
    imgs = np.empty((len(stems), INPUT, INPUT, BANDS), np.float32)
    labels = np.empty((len(stems), 1), np.float32)
    for i, stem in enumerate(stems):
        full = np.asarray(mm[index[stem]])[:, :, :BANDS]
        imgs[i] = cv2.resize(full, (INPUT, INPUT), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        labels[i] = 1.0 if cloud[stem] >= CLOUD_THRESH else 0.0
    del mm
    return (imgs, labels)

class DS(Dataset):

    def __init__(self, imgs, labels, augment=False):
        self.imgs, self.labels, self.aug = (imgs, labels, augment)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        img = self.imgs[i]
        if self.aug:
            if np.random.rand() < 0.5:
                img = img[:, ::-1]
            if np.random.rand() < 0.5:
                img = img[::-1, :]
            k = np.random.randint(4)
            if k:
                img = np.rot90(img, k)
            img = np.ascontiguousarray(img)
        return (torch.from_numpy(img).permute(2, 0, 1), torch.from_numpy(self.labels[i]))

def evaluate(model, loader):
    model.eval()
    P, Y = ([], [])
    with torch.no_grad():
        for x, y in loader:
            P.append(torch.sigmoid(model(x)).numpy().ravel())
            Y.append(y.numpy().ravel())
    p = np.concatenate(P)
    y = np.concatenate(Y).astype(int)
    pred = (p >= 0.5).astype(int)
    return (accuracy_score(y, pred), f1_score(y, pred, zero_division=0))

def fine_tune(model, tl, vl, epochs, lr, save_path):
    crit = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_f1 = 0.0
    for ep in range(epochs):
        model.train()
        ls = 0.0
        for x, y in tl:
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            ls += loss.item()
        acc, f1 = evaluate(model, vl)
        mk = ''
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), save_path)
            mk = '  +best'
        print(f'    ep {ep + 1:2d} loss={ls / len(tl):.4f} val_acc={acc:.4f} val_f1={f1:.4f}{mk}', flush=True)
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache_npy', default='/cache/splits_4band_384.npy')
    ap.add_argument('--cache_json', default='/cache/splits_4band_384_index.json')
    ap.add_argument('--splits_dir', default='/workspace')
    ap.add_argument('--weights', default='/workspace/encoder_96_4b.pth')
    ap.add_argument('--train_size', type=int, default=2000)
    ap.add_argument('--ratios', default='0.55,0.75')
    ap.add_argument('--epochs', type=int, default=25)
    ap.add_argument('--lr', type=float, default=0.0001)
    ap.add_argument('--scan', action='store_true', help='prune-only sweep, print params, no fine-tune')
    ap.add_argument('--out', default='/workspace/prune_results.json')
    a = ap.parse_args()
    torch.manual_seed(42)
    np.random.seed(42)
    rd = lambda n: [l.strip() for l in open(os.path.join(a.splits_dir, f'split_{n}.txt')) if l.strip()]
    train, val, test = (rd('train')[:a.train_size], rd('calib'), rd('test'))
    print(f'train={len(train)} val(calib)={len(val)} test={len(test)}  -- preloading to RAM...', flush=True)
    tr_x, tr_y = preload(a.cache_npy, a.cache_json, train)
    va_x, va_y = preload(a.cache_npy, a.cache_json, val)
    te_x, te_y = preload(a.cache_npy, a.cache_json, test)
    tl = DataLoader(DS(tr_x, tr_y, True), batch_size=8, shuffle=True, num_workers=0)
    vl = DataLoader(DS(va_x, va_y), batch_size=8, shuffle=False, num_workers=0)
    tel = DataLoader(DS(te_x, te_y), batch_size=8, shuffle=False, num_workers=0)
    base_state = torch.load(a.weights, map_location=DEVICE)
    bw = widths_of(base_state)
    model = Encoder4Band(widths=bw)
    model.load_state_dict(base_state)
    n_base = sum((p.numel() for p in model.parameters()))
    base_acc, base_f1 = evaluate(model, tel)
    print(f'Baseline widths={bw} params={n_base / 1000000.0:.4f}M  TEST acc={base_acc:.4f} f1={base_f1:.4f}')
    runner = IterativePruningRunner(model, torch.randn(1, BANDS, INPUT, INPUT))
    print('Running sensitivity analysis (ana)...', flush=True)
    t0 = time.time()
    runner.ana(lambda m: evaluate(m, vl)[0])
    print(f'  ana done ({(time.time() - t0) / 60:.1f} min)')
    if a.scan:
        print('\n[scan] ratio -> params (no fine-tune):')
        for r in [float(x) for x in a.ratios.split(',')]:
            slim = runner.prune(removal_ratio=r, mode='slim', channel_divisible=2)
            w = widths_of(slim.state_dict())
            n = sum((p.numel() for p in slim.parameters()))
            print(f'  r={r:<5} params={n / 1000000.0:.4f}M ({100 * n / n_base:4.1f}%) widths={w}', flush=True)
        return
    results = []
    for r in [float(x) for x in a.ratios.split(',')]:
        print(f'\n=== removal_ratio={r} ===', flush=True)
        slim = runner.prune(removal_ratio=r, mode='slim', channel_divisible=2)
        w = widths_of(slim.state_dict())
        n = sum((p.numel() for p in slim.parameters()))
        print(f'  pruned widths={w} params={n / 1000000.0:.4f}M ({100 * n / n_base:.1f}% of baseline)')
        save_path = f'/workspace/encoder_96_4b_prune_r{r}.pth'
        fine_tune(slim, tl, vl, a.epochs, a.lr, save_path)
        acc, f1 = evaluate(slim, tel)
        print(f'  FINAL r={r}: params={n / 1000000.0:.4f}M  TEST acc={acc:.4f} f1={f1:.4f}')
        results.append({'ratio': r, 'params_M': round(n / 1000000.0, 4), 'pct_of_baseline': round(100 * n / n_base, 1), 'widths': w, 'test_acc': round(acc, 4), 'test_f1': round(f1, 4), 'weights': os.path.basename(save_path)})
        json.dump({'baseline': {'widths': bw, 'params_M': round(n_base / 1000000.0, 4), 'test_acc': round(base_acc, 4)}, 'pruned': results}, open(a.out, 'w'), indent=2)
    print('\nDone:')
    for x in results:
        print(f"  r={x['ratio']} {x['params_M']}M acc={x['test_acc']} widths={x['widths']}")
if __name__ == '__main__':
    main()
