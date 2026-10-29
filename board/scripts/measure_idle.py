import time, json
from pynq import get_rails

def sample(secs, label):
    rails = get_rails()
    prails = [n for n, r in rails.items() if r.power is not None]
    acc = {n: [] for n in prails}
    t0 = time.time()
    while time.time() - t0 < secs:
        for n in prails:
            acc[n].append(rails[n].power.value)
        time.sleep(0.05)
    means = {n: sum(v) / len(v) for n, v in acc.items()}
    print(f'[{label}] total = {sum(means.values()):.3f} W   ({len(acc[prails[0]])} samples)')
    for n in sorted(means, key=lambda x: -means[x]):
        print(f'    {n:10s} {means[n]:.4f}')
    return means
print('=== idle BEFORE overlay (10 s) ===')
before = sample(10, 'before')
print('\n=== loading DPU overlay ===')
from pynq_dpu import DpuOverlay
ov = DpuOverlay('dpu.bit')
print('overlay loaded\n')
print('=== idle AFTER overlay (10 s) ===')
after = sample(10, 'after')
json.dump({'before': before, 'after': after, 'before_total': sum(before.values()), 'after_total': sum(after.values())}, open('/home/xilinx/TESTES_FINAIS/idle_baseline.json', 'w'), indent=2)
print(f'\ndelta from loading overlay: {sum(after.values()) - sum(before.values()):+.3f} W')
print('saved -> idle_baseline.json')
