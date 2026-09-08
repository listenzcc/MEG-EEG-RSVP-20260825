# %%
from util.easy_imports import *

# %%
DATA_DIR = Path('output/sliding-decode')

# %%
times = np.linspace(-0.5, 1.5, 401)

# %%
# EEG

MODES = ['EEG', 'MEG']

for _mode in MODES:
    scores = []
    for folder in DATA_DIR.glob(f'{_mode}-*'):
        mode, subj = folder.name.split('-')
        fname = folder / 'scores.txt'
        _scores = np.loadtxt(fname)
        scores.append(_scores)
    scores = np.vstack(scores)
    print(scores.shape)

    plt.plot(np.mean(scores, axis=0))
    plt.title(mode)
    plt.show()

