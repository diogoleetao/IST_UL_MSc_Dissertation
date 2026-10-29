import argparse
import numpy as np
from pynq_dpu import DpuOverlay

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--xmodel', required=True)
    ap.add_argument('--overlay', default='dpu.bit', help='DPU overlay .bit (default: stock B1600)')
    ap.add_argument('--n', type=int, default=100, help='number of inferences')
    args = ap.parse_args()
    overlay = DpuOverlay(args.overlay)
    overlay.load_model(args.xmodel)
    runner = overlay.runner
    in_t = runner.get_input_tensors()[0]
    out_t = runner.get_output_tensors()[0]
    in_buf = [np.zeros(tuple(in_t.dims), dtype=np.int8)]
    out_buf = [np.zeros(tuple(out_t.dims), dtype=np.int8)]
    for _ in range(args.n):
        job = runner.execute_async(in_buf, out_buf)
        runner.wait(job)
    print(f'ran {args.n} inferences on {args.xmodel}')
if __name__ == '__main__':
    main()
