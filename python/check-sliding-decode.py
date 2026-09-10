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

fig, axes = plt.subplots(1, 2, figsize=(12, 6))

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

    ax = axes[MODES.index(_mode)]
    ax.plot(
        times,
        np.mean(scores, axis=0),
        label=f'{_mode}-{_rma}-{_decod}',
        alpha=0.8)

for ax in axes:
    ax.legend()
    ax.set_ylim([0.4, 1.0])
    ax.axvline(0, color='k', linestyle='--', alpha=0.5)

fig.tight_layout()
plt.show()

# %%
