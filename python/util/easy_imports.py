# %%
import os
import sys
import mne
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from rich import print
from loguru import logger
from pathlib import Path
from tqdm.auto import tqdm
from itertools import product
from IPython.display import display

# %%
n_jobs = 32
logger.add('log/MEG-EEG-RSVP.log', rotation='1 MB')

# %%
