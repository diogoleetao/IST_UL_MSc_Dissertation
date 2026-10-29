import os, csv, sys, glob, re, argparse, subprocess
from pathlib import Path
WS = Path('/workspace')
ARCH = WS / 'pynqdpu_arch.json'
QUANT = WS / 'quant'
COMPILED = WS / 'compiled'
SUMMARY = WS / 'calib_compile_summary.csv'

def sh(cmd, logpath):
    with open(logpath, 'w') as f:
        return subprocess.run([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT).returncode

def drop_from_log(logpath):
    for line in Path(logpath).read_text(errors='ignore').splitlines():
        m = re.search('FP32 ([\\d.]+) \\| INT8 ([\\d.]+) \\| drop ([+\\-][\\d.]+)', line)
        if m:
            return (m.group(1), m.group(2), m.group(3))
    return ('', '', '')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--only', default=None, help='substring filter on tag')
    ap.add_argument('--cache_npy', default=None, help='pixcache .npy -> forwarded to calib (skips E: mount)')
    ap.add_argument('--cache_json', default=None)
    ap.add_argument('--fast', action='store_true', help='forward --fast to calib (calib+export only, ~3x faster)')
    a = ap.parse_args()
    cache_args = ['--cache_npy', a.cache_npy, '--cache_json', a.cache_json] if a.cache_npy else []
    if a.fast:
        cache_args.append('--fast')
    QUANT.mkdir(exist_ok=True)
    COMPILED.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(WS / 'train_results.csv')))
    if a.only:
        rows = [r for r in rows if a.only in r['tag']]
    new = not SUMMARY.exists()
    out = open(SUMMARY, 'a', newline='')
    w = csv.writer(out)
    if new:
        w.writerow(['tag', 'model', 'input', 'bands', 'width', 'fp32', 'int8', 'drop', 'calib_ok', 'compile_ok'])
    for r in rows:
        tag, model = (r['tag'], r['model'])
        inp, bands, width = (r['input'], r['bands'], r['width'])
        pth = WS / f'{tag}.pth'
        if not pth.exists():
            print(f'[skip] {tag}: no .pth', flush=True)
            continue
        if (COMPILED / tag / f'{tag}.xmodel').exists():
            print(f'[skip] {tag}: already compiled', flush=True)
            continue
        qd = QUANT / tag
        qd.mkdir(parents=True, exist_ok=True)
        rc1 = sh([sys.executable, WS / 'calib_generic.py', '--model', model, '--input', inp, '--bands', bands, '--width', width, '--weights', pth, '--output_dir', qd] + cache_args, qd / 'calib.log')
        ints = glob.glob(str(qd / '*_int.xmodel'))
        fp32, int8, drop = drop_from_log(qd / 'calib.log')
        calib_ok = rc1 == 0 and bool(ints)
        compile_ok = False
        if calib_ok:
            od = COMPILED / tag
            od.mkdir(parents=True, exist_ok=True)
            rc2 = sh(['vai_c_xir', '-x', ints[0], '-a', ARCH, '-o', od, '-n', tag], od / 'compile.log')
            compile_ok = rc2 == 0 and (od / f'{tag}.xmodel').exists()
        print(f"{tag:22s} calib={('ok' if calib_ok else 'FAIL'):4s} compile={('ok' if compile_ok else 'FAIL'):4s}  drop={drop}", flush=True)
        w.writerow([tag, model, inp, bands, width, fp32, int8, drop, int(calib_ok), int(compile_ok)])
        out.flush()
    out.close()
    print(f'\nDone -> {SUMMARY.name}  (xmodels in compiled/<tag>/<tag>.xmodel)')
if __name__ == '__main__':
    main()
