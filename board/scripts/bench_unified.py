import os
import sys
import csv
import time
import argparse
import json
import queue
import threading
import statistics
import numpy as np
from PIL import Image
try:
    import cv2
    HAVE_CV2 = True
except ImportError:
    HAVE_CV2 = False
from pynq_dpu import DpuOverlay
DECISION_THR = 0.5
PIXEL_THRESH = 0.5
CLOUD_THRESH = 0.3
PHASES = ('decode', 'resize', 'normalize', 'stack', 'quant', 'dpu', 'dequant', 'activation', 'decision')

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

class PowerSampler(threading.Thread):

    def __init__(self, period=0.05):
        super().__init__(daemon=True)
        self.period = period
        self.times = []
        self.rows = []
        self._stopflag = False
        self.rails = None
        try:
            from pynq import get_rails
            self.rails = get_rails()
            self.prails = [n for n, r in self.rails.items() if r.power is not None]
        except Exception as e:
            print(f'  [power] get_rails unavailable ({e}) -- power skipped')
            self.prails = []

    def run(self):
        if self.rails is None:
            return
        while not self._stopflag:
            self.times.append(time.perf_counter())
            self.rows.append({n: self.rails[n].power.value for n in self.prails})
            time.sleep(self.period)

    def stop(self):
        self._stopflag = True

    def _rows_in(self, t0, t1):
        return [row for t, row in zip(self.times, self.rows) if t0 <= t <= t1]

    def per_rail_mean(self, t0, t1):
        rows = self._rows_in(t0, t1)
        if not rows:
            return {n: 0.0 for n in self.prails}
        return {n: statistics.mean((r[n] for r in rows)) for n in self.prails}

    def per_rail_peak(self, t0, t1):
        rows = self._rows_in(t0, t1)
        if not rows:
            return {n: 0.0 for n in self.prails}
        return {n: max((r[n] for r in rows)) for n in self.prails}

    def total_window(self, t0, t1):
        return [sum(r.values()) for r in self._rows_in(t0, t1)]

def stats(xs):
    return (statistics.mean(xs), statistics.pstdev(xs) if len(xs) > 1 else 0.0, min(xs), max(xs))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xmodel', required=True)
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--split', default=None, help='file with one patch stem per line (e.g. split_clean_test.txt); default = first --max_patches files in train_red')
    ap.add_argument('--max_patches', type=int, default=200)
    ap.add_argument('--runs', type=int, default=10)
    ap.add_argument('--overlay', default='dpu.bit')
    ap.add_argument('--backend', choices=['cv2', 'pil'], default='cv2')
    ap.add_argument('--tag', default=None)
    ap.add_argument('--idle_json', default=None, help="fixed idle baseline (measure_idle.py output); uses the post-overlay 'after' idle instead of a per-model window")
    ap.add_argument('--parallel', action='store_true', help='pipelined mode: a producer thread pre-processes patch N+1 (CPU) while the DPU runs patch N. Measures the overlapped end-to-end latency and the power of both engines running together. Serial mode is the default.')
    ap.add_argument('--out', default='bench_unified.csv')
    args = ap.parse_args()
    tag = args.tag or os.path.splitext(os.path.basename(args.xmodel))[0]
    use_cv2 = args.backend == 'cv2' and HAVE_CV2
    overlay = DpuOverlay(args.overlay)
    overlay.load_model(args.xmodel)
    runner = overlay.runner
    in_t, out_t = (runner.get_input_tensors()[0], runner.get_output_tensors()[0])
    in_scale = 2 ** in_t.get_attr('fix_point')
    out_scale = 2 ** out_t.get_attr('fix_point')
    in_shape, out_shape = (tuple(in_t.dims), tuple(out_t.dims))
    size, bands = (in_shape[1], in_shape[3])
    out_elems = int(np.prod(out_shape[1:]))
    mtype = 'encoder' if out_elems <= 2 else 'FCN'
    print(f"[{tag}] in {in_shape} out {out_shape} | size={size} bands={bands} | {mtype} | backend={('cv2' if use_cv2 else 'pil')}")
    bandnames = ['red', 'green', 'blue'] + (['nir'] if bands == 4 else [])
    if args.split:
        stems = [l.strip() for l in open(args.split) if l.strip()]
    else:
        reds = sorted((f[len('red_'):] for f in os.listdir(os.path.join(args.data_dir, 'train_red')) if f.startswith('red_')))
        stems = reds[:args.max_patches]
    n = len(stems)
    paths = [[os.path.join(args.data_dir, f'train_{b}', f'{b}_{s}') for b in bandnames] for s in stems]
    gtpaths = [os.path.join(args.data_dir, 'train_gt', f'gt_{s}') for s in stems]
    in_buf = [np.zeros(in_shape, dtype=np.int8)]
    out_buf = [np.zeros(out_shape, dtype=np.int8)]

    def decode(p):
        if use_cv2:
            return cv2.imread(p, cv2.IMREAD_UNCHANGED)
        return np.array(Image.open(p), dtype=np.float32)

    def resize_one(a):
        if a.shape[:2] == (size, size):
            return a
        if use_cv2:
            return cv2.resize(a, (size, size), interpolation=cv2.INTER_AREA)
        return np.array(Image.fromarray(a).resize((size, size), Image.BILINEAR), dtype=np.float32)

    def one_run(collect_pred=False):
        acc = dict.fromkeys(PHASES + ('pre', 'post', 'cpu', 'total'), 0.0)
        preds = []
        for pp in paths:
            t0 = time.perf_counter()
            arrs = [decode(p) for p in pp]
            t1 = time.perf_counter()
            arrs = [resize_one(a) for a in arrs]
            t2 = time.perf_counter()
            arrs = [a.astype(np.float32) / 255.0 for a in arrs]
            t3 = time.perf_counter()
            img = np.stack(arrs, axis=-1)
            t4 = time.perf_counter()
            in_buf[0][0] = np.clip(np.round(img * in_scale), -128, 127).astype(np.int8)
            t5 = time.perf_counter()
            job = runner.execute_async(in_buf, out_buf)
            runner.wait(job)
            t6 = time.perf_counter()
            if mtype == 'encoder':
                logit = float(out_buf[0].astype(np.float32).flatten()[0]) / out_scale
                t7 = time.perf_counter()
                prob = sigmoid(logit)
                t8 = time.perf_counter()
                pred = 1 if prob >= DECISION_THR else 0
                t9 = time.perf_counter()
            else:
                logits = out_buf[0][0].astype(np.float32) / out_scale
                t7 = time.perf_counter()
                prob = sigmoid(logits).squeeze()
                t8 = time.perf_counter()
                frac = float((prob > PIXEL_THRESH).mean())
                pred = 1 if frac >= CLOUD_THRESH else 0
                t9 = time.perf_counter()
            acc['decode'] += (t1 - t0) * 1000.0
            acc['resize'] += (t2 - t1) * 1000.0
            acc['normalize'] += (t3 - t2) * 1000.0
            acc['stack'] += (t4 - t3) * 1000.0
            acc['quant'] += (t5 - t4) * 1000.0
            acc['dpu'] += (t6 - t5) * 1000.0
            acc['dequant'] += (t7 - t6) * 1000.0
            acc['activation'] += (t8 - t7) * 1000.0
            acc['decision'] += (t9 - t8) * 1000.0
            acc['total'] += (t9 - t0) * 1000.0
            if collect_pred:
                preds.append(pred)
        acc['pre'] = acc['decode'] + acc['resize'] + acc['normalize'] + acc['stack'] + acc['quant']
        acc['post'] = acc['dequant'] + acc['activation'] + acc['decision']
        acc['cpu'] = acc['pre'] + acc['post']
        return (acc, preds)

    def run_parallel(collect_pred=False):
        acc = dict.fromkeys(PHASES + ('pre', 'post', 'cpu', 'total'), 0.0)
        q = queue.Queue(maxsize=3)
        STOP = object()
        cpu_busy = [0.0]

        def producer():
            for pp in paths:
                t0 = time.perf_counter()
                arrs = [decode(p) for p in pp]
                arrs = [resize_one(a) for a in arrs]
                arrs = [a.astype(np.float32) / 255.0 for a in arrs]
                img = np.stack(arrs, axis=-1)
                q8 = np.clip(np.round(img * in_scale), -128, 127).astype(np.int8)
                cpu_busy[0] += (time.perf_counter() - t0) * 1000.0
                q.put(q8)
            q.put(STOP)
        preds, dpu_busy = ([], 0.0)
        t_wall0 = time.perf_counter()
        prod = threading.Thread(target=producer, daemon=True)
        prod.start()
        while True:
            q8 = q.get()
            if q8 is STOP:
                break
            in_buf[0][0] = q8
            td0 = time.perf_counter()
            job = runner.execute_async(in_buf, out_buf)
            runner.wait(job)
            dpu_busy += (time.perf_counter() - td0) * 1000.0
            if mtype == 'encoder':
                logit = float(out_buf[0].astype(np.float32).flatten()[0]) / out_scale
                pred = 1 if sigmoid(logit) >= DECISION_THR else 0
            else:
                logits = out_buf[0][0].astype(np.float32) / out_scale
                frac = float((sigmoid(logits) > PIXEL_THRESH).mean())
                pred = 1 if frac >= CLOUD_THRESH else 0
            if collect_pred:
                preds.append(pred)
        prod.join()
        acc['total'] = (time.perf_counter() - t_wall0) * 1000.0
        acc['cpu'] = acc['pre'] = cpu_busy[0]
        acc['dpu'] = dpu_busy
        return (acc, preds)
    runfn = run_parallel if args.parallel else one_run
    mode = 'parallel' if args.parallel else 'serial'
    print(f'Warmup over {n} patches ({mode} mode)...')
    runfn()
    sampler = PowerSampler()
    sampler.start()
    if args.idle_json:
        idle_rail = json.load(open(args.idle_json))['after']
        idle_total = sum(idle_rail.values())
        print(f'  [idle] fixed baseline from {args.idle_json}: {idle_total:.3f} W')
        time.sleep(0.5)
    else:
        time.sleep(2.5)
        t_idle0 = sampler.times[0] if sampler.times else 0.0
        t_idle1 = time.perf_counter()
        idle_rail = sampler.per_rail_mean(t_idle0, t_idle1)
        idle_total = sum(idle_rail.values())
    per_run = {k: [] for k in PHASES + ('pre', 'post', 'cpu', 'total')}
    t_run0 = time.perf_counter()
    preds_final = []
    for i in range(args.runs):
        acc, preds = runfn(collect_pred=i == 0)
        for k in per_run:
            per_run[k].append(acc[k] / n)
        if i == 0:
            preds_final = preds
        print(f"  run {i + 1}/{args.runs}: {acc['total'] / n:.3f} ms/patch", flush=True)
    t_run1 = time.perf_counter()
    sampler.stop()
    sampler.join(timeout=1.0)
    act_rail_mean = sampler.per_rail_mean(t_run0, t_run1)
    act_rail_peak = sampler.per_rail_peak(t_run0, t_run1)
    act_total = sum(act_rail_mean.values())
    tot_win = sampler.total_window(t_run0, t_run1)
    peak_total = max(tot_win) if tot_win else 0.0
    delta_total = act_total - idle_total
    delta_rail = {r: act_rail_mean[r] - idle_rail.get(r, 0.0) for r in act_rail_mean}
    mean_total = statistics.mean(per_run['total'])
    energy_pp = delta_total * (mean_total / 1000.0) * 1000.0
    prails = list(act_rail_mean.keys())
    correct = 0
    for pred, gp in zip(preds_final, gtpaths):
        gt = np.array(Image.open(gp), dtype=np.float32)
        gt_lab = 1 if (gt > 127).mean() >= CLOUD_THRESH else 0
        correct += pred == gt_lab
    accuracy = correct / n if n else 0.0
    dpu_power = delta_rail.get('INT', 0.0)
    cpu_power = delta_rail.get('PSINT_FP', 0.0)
    vals = {k: statistics.mean(per_run[k]) for k in per_run}
    _, dpu_sd, _, _ = stats(per_run['dpu'])
    _, cpu_sd, _, _ = stats(per_run['cpu'])
    _, tot_sd, _, _ = stats(per_run['total'])
    RAIL_ORDER = ['INT', 'PSINT_FP', 'PSDDR', '3V3', 'AUX', 'PSAUX', 'PSINT_LP', 'PSPLL', '1V2', '3V3_D']
    ordered = [r for r in RAIL_ORDER if r in prails] + [r for r in prails if r not in RAIL_ORDER]
    print(f"\n{'-' * 78}")
    print(f'  {tag}  ({mtype}, {size}x{size}, {bands}b)  [{mode}]  {args.runs} runs x {n} patches')
    print(f"{'-' * 78}")
    print(f"  accuracy   {accuracy:.4f}   time_total {vals['total']:.3f} ms/patch   {1000.0 / mean_total:.1f} patches/s")
    if mode == 'parallel':
        print(f"  [overlap]  cpu_busy {vals['pre']:.3f} | dpu_busy {vals['dpu']:.3f}  ms/patch  (wall = time_total above; serial would be ~cpu_busy+dpu_busy)")
    else:
        print(f"  time_pre {vals['pre']:.3f} | time_dpu {vals['dpu']:.3f} | time_post {vals['post']:.3f}  ms/patch")
    print(f'  board  mean {act_total:.3f} W  peak {peak_total:.3f} W  delta {delta_total:+.3f} W')
    print(f'  dpu_power(INT) {dpu_power:+.3f} W   cpu_power(PSINT_FP) {cpu_power:+.3f} W   energy {energy_pp:.3f} mJ/patch')
    print(f'  std: total {tot_sd:.3f}  dpu {dpu_sd:.3f}  cpu {cpu_sd:.3f}  ms/patch')
    print(f"  {'-' * 40}")
    print(f"  {'op':<11}{'mean':>9}{'std':>8}{'min':>9}{'max':>9}  ms/patch")
    for k in PHASES:
        m, sd, mn, mx = stats(per_run[k])
        print(f'  {k:<11}{m:9.3f}{sd:8.3f}{mn:9.3f}{mx:9.3f}')
    print(f"  {'-' * 40}")
    print(f'  power per rail (W)   idle -> mean / peak   delta')
    for r in ordered:
        print(f'    {r:10s} {idle_rail.get(r, 0.0):7.3f} -> {act_rail_mean[r]:7.3f} / {act_rail_peak[r]:7.3f}   {delta_rail[r]:+7.3f}')
    print(f"{'-' * 78}")
    PRE_OPS = ['decode', 'resize', 'normalize', 'stack', 'quant']
    POST_OPS = ['dequant', 'activation', 'decision']
    newfile = not os.path.exists(args.out)
    with open(args.out, 'a') as f:
        important = ['model', 'accuracy', 'time_total', 'throughput', 'time_pre', 'time_dpu', 'time_post', 'board_mean_W', 'board_peak_W', 'board_delta_W', 'dpu_power_W', 'cpu_power_W', 'energy_mJ', 'time_total_std_ms', 'dpu_std_ms', 'cpu_std_ms']
        meta = ['mode', 'img_lib', 'type', 'size', 'bands', 'runs', 'n_patches']
        perop = [f'time_{p}' for p in PRE_OPS + POST_OPS]
        rail_cols = []
        for r in ordered:
            rail_cols += [f'{r}_idle_W', f'{r}_mean_W', f'{r}_peak_W', f'{r}_delta_W']
        if newfile:
            f.write(','.join(important + meta + perop + rail_cols) + '\n')
        row = [tag, f'{accuracy:.4f}', f"{vals['total']:.4f}", f'{1000.0 / mean_total:.1f}', f"{vals['pre']:.4f}", f"{vals['dpu']:.4f}", f"{vals['post']:.4f}", f'{act_total:.3f}', f'{peak_total:.3f}', f'{delta_total:.3f}', f'{dpu_power:.4f}', f'{cpu_power:.4f}', f'{energy_pp:.3f}', f'{tot_sd:.4f}', f'{dpu_sd:.4f}', f'{cpu_sd:.4f}']
        row += [mode, args.backend, mtype, size, bands, args.runs, n]
        row += [f'{vals[p]:.4f}' for p in PRE_OPS + POST_OPS]
        for r in ordered:
            row += [f'{idle_rail.get(r, 0.0):.4f}', f'{act_rail_mean[r]:.4f}', f'{act_rail_peak[r]:.4f}', f'{delta_rail[r]:.4f}']
        f.write(','.join((str(x) for x in row)) + '\n')
    print(f'Saved -> {args.out}')
if __name__ == '__main__':
    main()
