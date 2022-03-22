from sklearn.manifold import TSNE
import numpy as np
from matplotlib import pyplot as plt
from data.image_folder import make_dataset
import os

np.random.seed(42)

dataroot = '../pre-processing/subset_classes_single'

resultsroot= './saved_results/five-classes/bear/latent_code/bear_256x512/test_latest/images'
outpath = './tsne-five.png'

# resultsroot= './saved_results/single-class/bear/latent_code/bear_256x512/test_latest/images'
# outpath = './tsne-single.png'




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

for idx, fname in enumerate(os.listdir(resultsroot)):
    fpath = os.path.join(resultsroot, fname)
    embedding = np.load(fpath)
    image_embeddings.append(embedding)
print("embeddings loaded")

# data = list(zip(class_names, image_embeddings, class_indices))
# np.random.shuffle(data)
# data = data[:embeddings_to_plot]
# class_names = [p[0] for p in data]
# image_embeddings = [p[1] for p in data]
# class_indices = [p[2] for p in data]


print("doing tsne...")
tsne_embedded = TSNE(n_components=2, n_jobs=32,
                     learning_rate="auto", init="pca").fit_transform(np.array([e.flatten() for e in image_embeddings]))
print("done")

fig = plt.figure(1, (17., 15.))
# plt.scatter(tsne_embedded[:, 0], tsne_embedded[:, 1], alpha=1.0, s=2,
#             c=[v for v in class_indices], cmap="plasma")
scatter = plt.scatter(tsne_embedded[:, 0], tsne_embedded[:, 1], alpha=1.0, s=2,
            c=[v for v in class_indices], cmap="plasma", label=list(name2idx.keys()))
plt.legend(loc='upper right', handles=scatter.legend_elements()[0], labels=name2idx.keys())
# # cb = plt.colorbar()
# cb.set_label('Category')
plt.savefig(outpath)