from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from data.image_folder import make_dataset
from data.visamm_dataset import make_mm_dataset
import os
import torch
from torch import nn

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

class_indices = []
class_names = []
image_embeddings = []

for idx, path in enumerate(paths):
    classname = path.split('/')[-3]
    class_indices.append(name2idx[classname])
    class_names.append(classname)
print("classes loaded")

result_fnames = os.listdir(resultsroot)
result_fnames.sort(key=lambda p: int(p.split("_")[0]))
for idx, fname in enumerate(result_fnames):
    fpath = os.path.join(resultsroot, fname)
    embedding = np.load(fpath)
    image_embeddings.append(embedding)
print("embeddings loaded")

data = list(zip(class_names, image_embeddings, class_indices))
data.sort(key=lambda p: p[2])
class_names = [p[0] for p in data]
image_embeddings = torch.tensor([p[1].flatten() for p in data])
class_indices = [p[2] for p in data]

counts = []
curr_count = 0
prev_idx = 1
for idx in class_indices:
    if idx != prev_idx:
        prev_idx = idx
        counts.append(curr_count)
        curr_count = 0 
    else:
        curr_count += 1
counts.append(curr_count)

start = 0
print(counts)
for idx, count in enumerate(counts):
    
    cat_centroid = torch.tensor(image_embeddings[start: start + count].mean(axis=0))
    intra_cat_sim = nn.CosineSimilarity()(cat_centroid.expand(count, -1), image_embeddings[start: start + count])
    inter_cat_sim_1 = nn.CosineSimilarity()(cat_centroid.expand(start, -1), image_embeddings[0:start])
    inter_cat_sim_2 = nn.CosineSimilarity()(cat_centroid.expand(image_embeddings.shape[0]-count-start, -1),
                                            image_embeddings[start + count:])
    inter_cat_sim = torch.cat([inter_cat_sim_1, inter_cat_sim_2])
    
    print(f"{idx + 1}, {intra_cat_sim.mean().item():.4f}, {intra_cat_sim.std().item():.4f}, {inter_cat_sim.mean().item():.4f}, {inter_cat_sim.std().item():.4f}")
    start += count
