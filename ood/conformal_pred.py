"""
check if test slides are o.o.d using conformal pred
"""

import utils
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

dir_in = 'code/testing/'
model_path = 'model_perf_r34_b32'

# load model
if torch.cuda.is_available():
    loaded_model = torch.load(dir_in + model_path + '.pt')
