import cv2
import csv
import os
import numpy as np
try:
    import ctypes
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
DATA = 'E:\\Tese\\38_Cloud\\38-Cloud_training'
LOG = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'display_log.csv')
STEMS = ['patch_274_14_by_14_LC08_L1TP_039034_20160320_20170224_01_T1.TIF', 'patch_451_20_by_14_LC08_L1TP_059014_20160620_20170221_01_T1.TIF', 'patch_45_3_by_5_LC08_L1TP_002054_20160520_20170324_01_T1.TIF', 'patch_183_9_by_15_LC08_L1TP_045026_20160720_20170221_01_T1.TIF', 'patch_330_16_by_15_LC08_L1TP_029040_20160720_20170222_01_T1.TIF', 'patch_301_15_by_7_LC08_L1TP_029040_20160720_20170222_01_T1.TIF', 'patch_287_14_by_14_LC08_L1TP_034034_20160520_20170223_01_T1.TIF', 'patch_346_17_by_10_LC08_L1TP_029040_20160720_20170222_01_T1.TIF', 'patch_353_18_by_13_LC08_L1TP_002053_20160520_20170324_01_T1.TIF', 'patch_124_7_by_4_LC08_L1TP_039034_20160320_20170224_01_T1.TIF', 'patch_207_10_by_18_LC08_L1TP_034034_20160520_20170223_01_T1.TIF', 'patch_154_8_by_14_LC08_L1TP_035034_20160120_20170224_01_T1.TIF', 'patch_194_10_by_5_LC08_L1TP_034034_20160520_20170223_01_T1.TIF', 'patch_190_10_by_10_LC08_L1TP_035034_20160120_20170224_01_T1.TIF', 'patch_86_5_by_6_LC08_L1TP_039034_20160320_20170224_01_T1.TIF', 'patch_282_15_by_2_LC08_L1TP_002054_20160520_20170324_01_T1.TIF', 'patch_289_14_by_16_LC08_L1TP_011247_20160620_20170323_01_T1.TIF', 'patch_450_20_by_13_LC08_L1TP_059014_20160620_20170221_01_T1.TIF', 'patch_228_12_by_8_LC08_L1TP_035034_20160120_20170224_01_T1.TIF', 'patch_451_20_by_14_LC08_L1TP_044010_20160220_20170224_01_T1.TIF']
STEP, MIN_SQ = (40, 120)

def preview_bgr(stem):
    ch = {}
    for b in ('red', 'green', 'blue'):
        a = cv2.imread(f'{DATA}\\train_{b}\\{b}_{stem}', cv2.IMREAD_UNCHANGED)
        if a is None:
            raise FileNotFoundError(f'{DATA}\\train_{b}\\{b}_{stem}')
        a = a.astype(np.float32)
        lo, hi = (np.percentile(a, 2), np.percentile(a, 98))
        ch[b] = np.clip((a - lo) / max(hi - lo, 1.0) * 255, 0, 255).astype(np.uint8)
    return np.stack([ch['blue'], ch['green'], ch['red']], axis=-1)

def screen_size():
    try:
        import tkinter as tk
        r = tk.Tk()
        w, h = (r.winfo_screenwidth(), r.winfo_screenheight())
        r.destroy()
        return (w, h)
    except Exception:
        return (1920, 1080)

def main():
    sw, sh = screen_size()
    max_sq = min(sw, sh)
    sq = min(500, max_sq)
    win = 'patch'
    cv2.namedWindow(win, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    print(f'screen {sw}x{sh}  |  +/- resize, m = log size, space/a = next/prev, q = quit')
    i = 0
    while True:
        src = preview_bgr(STEMS[i])
        interp = cv2.INTER_AREA if sq < src.shape[0] else cv2.INTER_LINEAR
        img = cv2.resize(src, (sq, sq), interpolation=interp)
        canvas = np.zeros((sh, sw, 3), np.uint8)
        y, x = ((sh - sq) // 2, (sw - sq) // 2)
        canvas[y:y + sq, x:x + sq] = img
        cv2.putText(canvas, f'{i + 1}/{len(STEMS)}  img {sq}x{sq}px  (src {src.shape[0]})  screen {sw}x{sh}', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.setWindowTitle(win, f'{i + 1}/{len(STEMS)}  img {sq}px  {STEMS[i]}')
        cv2.imshow(win, canvas)
        k = cv2.waitKey(0) & 255
        if k in (ord('q'), 27):
            break
        elif k in (ord('d'), ord(' '), 83):
            i = (i + 1) % len(STEMS)
        elif k in (ord('a'), 81):
            i = (i - 1) % len(STEMS)
        elif k in (ord('+'), ord('=')):
            sq = min(max_sq, sq + STEP)
        elif k in (ord('-'), ord('_')):
            sq = max(MIN_SQ, sq - STEP)
        elif k == ord('m'):
            try:
                _, _, ww, wh = cv2.getWindowImageRect(win)
            except Exception:
                ww, wh = (sw, sh)
            new = not os.path.exists(LOG)
            with open(LOG, 'a', newline='') as f:
                w = csv.writer(f)
                if new:
                    w.writerow(['idx', 'stem', 'source_px', 'display_px', 'window_w', 'window_h', 'monitor_w', 'monitor_h'])
                w.writerow([i + 1, STEMS[i], src.shape[0], sq, ww, wh, sw, sh])
            print(f'  logged  patch {i + 1}/{len(STEMS)}: src {src.shape[0]} -> display {sq}px | window {ww}x{wh} | monitor {sw}x{sh}  -> {LOG}')
    cv2.destroyAllWindows()
if __name__ == '__main__':
    main()
