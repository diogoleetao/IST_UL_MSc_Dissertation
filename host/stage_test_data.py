import os, shutil

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = '38-Cloud_training'
SPLIT = os.path.join(HERE, 'split_test.txt')
OUT = os.path.join(HERE, 'test_data')
BANDS = ['red', 'green', 'blue', 'nir', 'gt']

stems = [l.strip() for l in open(SPLIT) if l.strip()]
for b in BANDS:
    os.makedirs(os.path.join(OUT, f'train_{b}'), exist_ok=True)
for s in stems:
    for b in BANDS:
        shutil.copyfile(os.path.join(DATA_DIR, f'train_{b}', f'{b}_{s}'),
                        os.path.join(OUT, f'train_{b}', f'{b}_{s}'))
print(f'copied {len(stems)} patches ({len(BANDS)} bands each) -> {OUT}')
