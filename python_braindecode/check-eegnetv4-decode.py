# %%
from util.easy_imports import *

# %%
DATA_DIR = Path('output/neuralnetwork-decode')

# IPython
try:
    __IPYTHON__
    DATA_DIR = Path('../output/neuralnetwork-decode')
except:
    pass

# %%
buffer = []
for folder in DATA_DIR.iterdir():
    if not folder.is_dir():
        continue

    mode, subj, decod = folder.name.split('-')

    if (folder / 'metrics.json').exists():
        with open(folder / 'metrics.json', 'r') as f:
            metrics = json.load(f)
        print(metrics)
        buffer.append(metrics)

df = pd.DataFrame(buffer)
print(df)

group = df.groupby(['mode', 'decoding_method']).agg(
    {'balanced_accuracy': ['mean', 'std']})
print(group)

sns.boxplot(data=df, x='mode', y='balanced_accuracy', hue='decoding_method')
plt.show()

# %%
