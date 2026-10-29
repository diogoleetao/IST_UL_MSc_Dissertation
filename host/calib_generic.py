import os, sys, json, argparse
import numpy as np
import cv2
import torch
from pytorch_nndct.apis import torch_quantizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models_thesis import build_model, Encoder
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval

def fold_bn(m):
    import copy
    mf = copy.deepcopy(m).eval()
    for name, child in mf.named_children():
        if isinstance(child, nn.Sequential):
            L = list(child)
            for i in range(len(L) - 1):
                if isinstance(L[i], nn.Conv2d) and isinstance(L[i + 1], nn.BatchNorm2d):
                    L[i] = fuse_conv_bn_eval(L[i], L[i + 1])
                    L[i + 1] = nn.Identity()
            setattr(mf, name, nn.Sequential(*L))
    return mf
PATCH_CROP, CLOUD_THRESH, PIX = (384, 30.0, 127)
CACHE = {}

def crop384(a):
    h, w = a.shape[:2]
    sh, sw = ((h - PATCH_CROP) // 2, (w - PATCH_CROP) // 2)
    return a[sh:sh + PATCH_CROP, sw:sw + PATCH_CROP]

def load_patch(data_dir, stem, bands, size):
    names = ['red', 'green', 'blue'] + (['nir'] if bands == 4 else [])
    ch = []
    for b in names:
        a = cv2.imread(os.path.join(data_dir, f'train_{b}', f'{b}_{stem}'), cv2.IMREAD_UNCHANGED)
        ch.append(crop384(a))
    img = np.stack(ch, -1)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    if img.ndim == 2:
        img = img[..., None]
    gt = crop384(cv2.imread(os.path.join(data_dir, 'train_gt', f'gt_{stem}'), cv2.IMREAD_UNCHANGED))
    label = 1 if (gt > PIX).mean() >= CLOUD_THRESH / 100.0 else 0
    return (torch.from_numpy(img).permute(2, 0, 1), label)

def read_split(splits_dir, name):
    return [l.strip() for l in open(os.path.join(splits_dir, f'split_{name}.txt')) if l.strip()]

def preload_cache(cache_npy, cache_json, stems, bands, size):
    meta = json.load(open(cache_json))
    index, cloud = (meta['index'], meta['cloud'])
    mm = np.load(cache_npy, mmap_mode='r')
    for stem in stems:
        full = np.asarray(mm[index[stem]])[:, :, :bands]
        img = cv2.resize(full, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
        if img.ndim == 2:
            img = img[..., None]
        label = 1 if cloud[stem] >= CLOUD_THRESH else 0
        CACHE[stem] = (torch.from_numpy(img).permute(2, 0, 1), label)
    del mm

def get_patch(data_dir, stem, bands, size):
    if stem in CACHE:
        return CACHE[stem]
    return load_patch(data_dir, stem, bands, size)

def decisions(model, stems, data_dir, bands, size, seg):
    model.eval()
    preds, gts = ([], [])
    with torch.no_grad():
        for stem in stems:
            x, gt = get_patch(data_dir, stem, bands, size)
            out = model(x.unsqueeze(0))
            if seg:
                frac = (torch.sigmoid(out) > 0.5).float().mean().item()
                preds.append(1 if frac >= CLOUD_THRESH / 100.0 else 0)
            else:
                preds.append(1 if torch.sigmoid(out).item() >= 0.5 else 0)
            gts.append(gt)
    p, g = (np.array(preds), np.array(gts))
    return float((p == g).mean())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['encoder', 'fcn', 'faithful', 'encoder_faithful'])
    ap.add_argument('--input', type=int, required=True)
    ap.add_argument('--bands', type=int, required=True, choices=[3, 4])
    ap.add_argument('--width', type=float, default=1.0)
    ap.add_argument('--weights', required=True)
    ap.add_argument('--output_dir', required=True)
    ap.add_argument('--data_dir', default='/data/38_Cloud/38-Cloud_training')
    ap.add_argument('--splits_dir', default='/workspace')
    ap.add_argument('--cache_npy', default=None, help='pixcache .npy (skips the E: dataset mount)')
    ap.add_argument('--cache_json', default=None)
    ap.add_argument('--fast', action='store_true', help='skip the FP32 eval (already in train_results.csv) and the INT8-sim eval (the board measures real INT8) -> only calib + export. ~3x faster.')
    ap.add_argument('--fold', action='store_true', help='pre-fold Conv+BN before calibration (BN-folding structure check): the resulting xmodel should be identical to compiling the un-folded model.')
    a = ap.parse_args()
    seg = a.model in ('fcn', 'faithful')
    os.makedirs(a.output_dir, exist_ok=True)
    dev = torch.device('cpu')
    st = torch.load(a.weights, map_location=dev)
    if isinstance(st, dict) and 'state_dict' in st:
        st = st['state_dict']
    pruned_w = None
    if a.model == 'encoder':
        stage_w = [int(st[f'enc{i}.0.weight'].shape[0]) for i in range(1, 6)]
        uniform = [max(8, int(round(w * a.width))) for w in (32, 64, 128, 256, 512)]
        if stage_w != uniform:
            pruned_w = stage_w
    model = Encoder(a.bands, widths=pruned_w) if pruned_w else build_model(a.model, a.bands, a.width)
    model.load_state_dict(st)
    model.eval()
    if a.fold:
        model = fold_bn(model)
        print('  [--fold] Conv+BN pre-folded (BN layers absorbed into convs)')
    wdesc = f'widths={pruned_w}' if pruned_w else f'width={a.width}'
    print(f'model={a.model} input={a.input} bands={a.bands} {wdesc} params={sum((p.numel() for p in model.parameters())) / 1000000.0:.4f}M')
    calib = read_split(a.splits_dir, 'calib')
    test = read_split(a.splits_dir, 'test')
    if a.cache_npy:
        print(f'preloading {len(calib) + len(test)} patches from pixcache (no E: mount)...')
        preload_cache(a.cache_npy, a.cache_json, calib + test, a.bands, a.input)
    dummy = torch.randn(1, a.bands, a.input, a.input)
    if not a.fast:
        print('\n[1/4] FP32 accuracy on TEST (clean, separate)...')
        fp32 = decisions(model, test, a.data_dir, a.bands, a.input, seg)
        print(f'  FP32 test acc: {fp32:.4f}')
    print('\n[2/4] INT8 calibration on split_calib...')
    q = torch_quantizer('calib', model, dummy, output_dir=a.output_dir, device=dev)
    qm = q.quant_model
    with torch.no_grad():
        for stem in calib:
            x, _ = get_patch(a.data_dir, stem, a.bands, a.input)
            qm(x.unsqueeze(0))
    q.export_quant_config()
    print(f'  quant_info.json -> {a.output_dir}')
    print('\n[3/4] build test quantizer...')
    q2 = torch_quantizer('test', model, dummy, output_dir=a.output_dir, device=dev)
    if a.fast:
        with torch.no_grad():
            x, _ = get_patch(a.data_dir, test[0], a.bands, a.input)
            q2.quant_model(x.unsqueeze(0))
        print('  (fast) skipped FP32/INT8 accuracy eval')
    else:
        int8 = decisions(q2.quant_model, test, a.data_dir, a.bands, a.input, seg)
        print(f'  FP32 {fp32:.4f} | INT8 {int8:.4f} | drop {fp32 - int8:+.4f}  (on split_test, separate)')
    print('\n[4/4] export xmodel...')
    q2.export_xmodel(output_dir=a.output_dir, deploy_check=True)
    print(f'  xmodel -> {a.output_dir}')
if __name__ == '__main__':
    main()
