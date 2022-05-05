from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import numpy as np
from matplotlib import pyplot as plt
from data.image_folder import make_dataset
from data.visamm_dataset import make_mm_dataset
import os
import torch
from torch import nn
import clip

from PIL import Image
from tqdm import tqdm

np.random.seed(42)


IS_MM = False

# dataroot = '../pre-processing/subset_classes'
dataroot = '../pre-processing/mmdata'

# resultsroot= './saved_results/five-classes/bear/latent_code/bear_256x512/test_latest/images'
# resultsroot= './results/debug/latent_code/debug/test_latest/images'
resultsroot= './results/visamm-gpt-five/latent_code/visamm-gpt-five/test_latest/images'

device = "cuda"


# if IS_MM:
#     paths = sorted(make_mm_dataset(dataroot, 1), key=lambda d: d["im_path"])
#     paths = [p["im_path"] for p in paths]
#     texts = [p["transc"] for p in paths]
# else:
#     paths = sorted(make_dataset(dataroot))

paths = sorted(make_mm_dataset(dataroot, 1), key=lambda d: d["im_path"])
img_paths, transc_paths = [p["im_path"] for p in paths], [p["transc_path"] for p in paths]


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

# Loading classes
for idx, path in enumerate(img_paths):
    classname = path.split('/')[-3]
    class_indices.append(name2idx[classname])
    class_names.append(classname)
print("classes loaded")

# Loading embeddings
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

# Loading CLIP labels as ground truth
clip_model, preprocess = clip.load("ViT-B/32", device=device)
clip_text_embeddings = []
texts = []
for transc_path in tqdm(transc_paths):
    with open(transc_path) as f:
        transc = f.readline().strip()[:77]
        texts.append(clip.tokenize(transc))
texts = torch.cat(texts).to(device)

with torch.no_grad():
    for txt in tqdm(texts):
        clip_text_embeddings.append(clip_model.encode_text(txt.unsqueeze(0)).detach().cpu())
clip_text_embeddings = torch.cat(clip_text_embeddings)
clip_text_embeddings = clip_text_embeddings.cpu().numpy()
print("clip embeddings loaded")

# Find nearest neighbors for each training data
N_NEIGHBORS = 10
knn_visa = NearestNeighbors(n_neighbors=N_NEIGHBORS, n_jobs=-1)
knn_visa.fit(image_embeddings)

_, neighbors_visa = knn_visa.kneighbors(image_embeddings) 

# Find nearest neighbors for each clip embedding
knn_clip = NearestNeighbors(n_neighbors=N_NEIGHBORS, n_jobs=-1)
knn_clip.fit(clip_text_embeddings)

_, neighbors_clip = knn_clip.kneighbors(clip_text_embeddings) 
metric = nn.CosineSimilarity()


# Count shared nearest neighbors
"""
for each produced embedding a, clip embedding b
    for each of a's 5 neighbors, b's 5 neighbors
        overlap += overlap

"""
total = 0
overlaps = []
for nei_visa, nei_clip, in zip(neighbors_visa, neighbors_clip):
    val = np.sum(np.in1d(nei_visa, nei_clip))
    total += val
    overlaps.append(val)

print("Total number of samples {}".format(len(neighbors_visa)))
print("Total neighbor overlaps: {} / {}".format(total, len(neighbors_visa) / N_NEIGHBORS))
print("Overlap rate: {}".format(total / len(neighbors_visa) / N_NEIGHBORS))
print("Std of number of overlaps: {}".format(np.std(overlaps)))
print(overlaps)


## Evaluate whether two points close in produced embedding are actually close in CLIP embedding space