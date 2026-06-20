import torch
import numpy as np
import pandas as pd
path = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/data/nmt/embeddings/prot_bert"
df = pd.read_csv("/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/data/nmt/nmt.csv")
emb = torch.load("%s/embeddings.pt" % path)
indices = torch.load("%s/indices.pt" % path)
y_values = torch.load("%s/y_values.pt" % path)
all_muts = df["num_muts"].values
muts = np.unique(all_muts)
for mut in muts:
    indices_mut = indices[all_muts == mut]
    y_values_mut = y_values[indices_mut]
    emb_mut = emb[indices_mut]
    torch.save(emb_mut, "%s/embeddings_of_nmut_%d.pt" % (path, mut))
    torch.save(indices_mut, "%s/indices_of_nmut_%d.pt" % (path, mut))
    torch.save(y_values_mut, "%s/y_values_of_nmut_%d.pt" % (path, mut))