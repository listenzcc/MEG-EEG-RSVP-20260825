# %%
from itertools import product
from util.easy_imports import *

# %%
DATA_DIR = Path('output/sliding-decode')

# IPython
DATA_DIR = Path('../output/sliding-decode')

# %%
times = np.linspace(-0.5, 1.5, 401)

# %%

MODES = ['EEG', 'MEG']
RMA = ['withoutRMA', 'withRMA']
DECOD = ['SVC', 'LR']

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

for _mode, _rma, _decod in product(MODES, RMA, DECOD):
    scores = []
    for folder in DATA_DIR.glob(f'{_mode}-*-{_decod}'):
        mode, subj, decod = folder.name.split('-')
        if _rma == 'withRMA':
            fname = folder / 'scores-rma.txt'
        else:
            fname = folder / 'scores.txt'

        if not fname.exists():
            print(f'File not found: {fname}')
            continue

        _scores = np.loadtxt(fname)
        scores.append(_scores)

    if not scores:
        print(f'No scores found for {_mode}-{_rma}-{_decod}')
        continue

    scores = np.vstack(scores)
    print(scores.shape)

    ax.plot(
        times,
        np.mean(scores, axis=0),
        label=f'{_mode}-{_rma}-{_decod}',
        alpha=0.8)

ax.legend()
plt.show()

# %%
