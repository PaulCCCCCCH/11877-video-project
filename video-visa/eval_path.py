from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.decomposition import PCA
import numpy as np
from matplotlib import pyplot as plt
from data.image_folder import make_dataset
from data.visamm_dataset import make_mm_dataset
import os
import torch
from torch import nn
from tqdm import tqdm

np.random.seed(42)


IS_MM = False

dataroot = '../pre-processing/subset_classes'
# dataroot = '../pre-processing/mmdata'

resultsroot= './saved_results/five-classes/bear/latent_code/bear_256x512/test_latest/images'
# resultsroot= './results/debug/latent_code/debug/test_latest/images'


if IS_MM:
    paths = sorted(make_mm_dataset(dataroot, 1), key=lambda d: d["im_path"])
    paths = [p["im_path"] for p in paths]
else:
    paths = sorted(make_dataset(dataroot))

# embeddings_to_plot = 500

name2idx = {
    'bicycling': 1,    
    'drawing': 2,
    'driving': 3,
    'running': 4,
    'surfing': 5
}

frame_embeddings = []
curr_frames = []
curr_frame_id = 0
result_fnames = os.listdir(resultsroot)
result_fnames.sort(key=lambda p: int(p.split("_")[0]))
for idx, fname in enumerate(tqdm(result_fnames[:200])):
# for idx, fname in enumerate(tqdm(result_fnames)):
    parts = fname.split("_")
    frame_id = int(parts[1])
    if frame_id < curr_frame_id:
        # assert curr_frame_id == 6, "Some videos have != 6 frames"
        # if curr_frame_id != 

        frame_embeddings.append(curr_frames)
        curr_frames = []
    curr_frame_id = frame_id
    fpath = os.path.join(resultsroot, fname)
    embedding = np.load(fpath)
    curr_frames.append(embedding)

print("frame embeddings loaded")


X = [emb[:-1] for emb in frame_embeddings] # frames 1 .. t-1 as X
Y = [emb[-1] for emb in frame_embeddings] # frame t as Y

# kernel = DotProduct() + WhiteKernel() 

for idx in range(10):
    x = X[idx] 
    y = Y[idx]
    x = np.array(x).reshape(len(x), -1) # x.shape: (5 frames, 24576 feature dim)
    y = np.array(y).flatten().reshape(1, -1) # y.shape: (1 frame, 24576 feature dim)

    pca = PCA(2)
    pca.fit(x)
    newx = pca.transform(x)
    newy = pca.transform(y)

    # DEBUG
    print(newx)
    print(newy)
    print("**************")

# gpr = GaussianProcessRegressor().fit(x, y)