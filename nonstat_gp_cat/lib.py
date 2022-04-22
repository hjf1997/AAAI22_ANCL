from config import Config
import numpy as np
import sys
import pandas as pd
import torch
from time import time
import warnings
warnings.filterwarnings('ignore')
from torch.cuda.amp import GradScaler, autocast
import torch.autograd.profiler as profiler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
# from snsgp import SNSGP
# from nsgp import NSGP
from nsgp_sgd import NSGP
# from nsgpytorch.utils.inducing import f_kmeans, f_random
import matplotlib.pyplot as plt
from IPython.display import clear_output
torch.autograd.set_detect_anomaly(True)


from sklearn.cluster import KMeans
import torch
import numpy as np


def f_kmeans(X, num_inducing_points, random_state=None):
    model = KMeans(n_clusters=num_inducing_points,
                   random_state=random_state)
    out = model.fit(X).cluster_centers_
    return torch.tensor(out, dtype=X.dtype)


def f_random(X, num_inducing_points, random_state=None):
    np.random.seed(random_state)
    inds = np.random.choice(
        X.shape[0], replace=False, size=num_inducing_points)
    return torch.tensor(X[inds], dtype=X.dtype)