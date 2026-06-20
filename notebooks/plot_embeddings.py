import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import TSNE

from utils_for_analysis import (
    DATASET_PATHS,
    positions
    # all other needed imports should be here if more are needed
)

# ==========================
# SETTINGS
# ==========================

DATASET_SETTINGS = {
    "lov": {
        "contour_tsne_kwargs": {
            "bandwidth": 2,
            "k": 50,
            "support_quantile": 99,
            "levels": 25,  # as in notebook
            "pad": 0.5,
            "scatter": True,
            "fill": True,
            "text_on_contour": False,
            "grid_jump": 0.25,
            "legend": False,
        },
        "tsne_kwargs": {
            "n_components": 2,
            "perplexity": 500,
            "learning_rate": 50,
            "random_state": 42,
            "init": "pca",
        },
        "embed": True,
        "sample": True,
        "n_samples": 15000,
        "K": 100,
        # indices to use for best, worst, pssm, esm8m
        "best": [690, 138, 571, 359, 17],
        "worst": [500, 537, 300, 723, 383],
        "pssm": [116, 368, 498, 523, 13],
        "esm8m": [13, 343, 729, 350, 10],
    },
    "gcn4": {
        "contour_tsne_kwargs": {
            "bandwidth": 1,
            "k": 50,
            "support_quantile": 99,
            "levels": 25,
            "pad": 0.5,
            "scatter": True,
            "fill": True,
            "text_on_contour": False,
            "grid_jump": 0.25,
            #"gof_threshold": 0.7,
            #"legend": False,
        },
        "tsne_kwargs": {
            "n_components": 2,
            "perplexity": 60,
            "learning_rate": 50,
            "random_state": 42,
            "init": "pca",
        },
        "embed": False,
        "sample": False,
        "K": 20,
        # indices to use for best, worst, pssm, esm8m (see notebook tsne_embeddings.ipynb lines 254+)
        "best": [690, 138, 571, 359, 17],
        "worst": [500, 537, 300, 723, 383],
        "pssm": [116, 368, 498, 523, 13],
        "esm8m": [13, 343, 729, 350, 10],
    },
    "pard3": {
        "contour_tsne_kwargs": {
            "bandwidth": 1.2,
            "k": 50,
            "support_quantile": 99,
            "levels": 20,
            "pad": 0.5,
            "scatter": True,
            "fill": True,
            "text_on_contour": False,
            "grid_jump": 0.25,
            #"legend": False,
        },
        "tsne_kwargs": {
            "n_components": 2,
            "perplexity": 60,
            "learning_rate": 50,
            "random_state": 42,
            "init": "pca",
        },
        "embed": False,
        "sample": False,
        "K": 50,
        # indices to use for best, worst, pssm, esm8m
        "best": [198, 134, 29, 143, 89],
        "worst":[192, 135, 58, 10, 119],
        "pssm": [194, 104, 143, 134],
        "esm8m": [89, 74, 194, 160],
    },
    # add other datasets as needed
}

def get_relevant_columns_gfp_protgym(df, first_col, last_col):
    si = np.where(df.columns == first_col)[0][0]
    ei = np.where(df.columns == last_col)[0][0]+1
    return df.columns[si:ei]


def get_relevant_columns(dataset, df):
    cols = get_relevant_columns_gfp_protgym(df, positions[dataset][0], positions[dataset][1])
    return cols


def get_one_hot_encoding(sdf, relevant_columns):
    one_hot_encoding = pd.get_dummies(sdf[relevant_columns])
    return one_hot_encoding



def embed_tsne(
    dataset_to_use, merged_indices, tsne_kwargs, output_path
):
    """
    Embed sequences using t-SNE and save the embeddings to a file.
    """
    embs = torch.load(
        f"/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/data/{dataset_to_use}/embeddings/esm_8m/embeddings.pt"
    )
    embs = embs[merged_indices]
    mean_embs = embs.mean(axis=1)
    mean_embs = (mean_embs - mean_embs.mean(axis=0)) / mean_embs.std(axis=0)
    tsne = TSNE(**tsne_kwargs)
    embs_2d = tsne.fit_transform(mean_embs)
    np.save(output_path, embs_2d)
    print(f"Saved TSNE embeddings to {output_path}")
    return embs_2d

def contour_tsne_values(
    tsne,
    z,
    bandwidth=1.0,
    k=8,
    support_quantile=95,
    levels=12,
    pad=0.5,
    scatter=False,
    scatter_indices=None,
    fill=True,
    text_on_contour=True,
    grid_jump=0.5,
    axes=None,
    ax_idx=None,
    legend=False,
    scatter_label="Special",
    contour_label=None,
    arrow_annotation=True, # add a label for the contourf if desired
):
    """
    tsne: array of shape (n, 2)
    z:    array of shape (n,)
    axes: matplotlib axes array, e.g., from plt.subplots(). If provided, use as axes.
    ax_idx: int index into axes array for which subplot to use.
    text_on_contour: if True, write numbers on contour lines, else just draw contours without labels
    grid_jump: Step size between grid points in each axis (default: 0.5)
    legend: bool, if True add a legend to the plot (for scatter, with scatter_label)
    scatter_label: str, label for the scatter points in the legend
    contour_label: str or None, label for the contourf if you want to show in legend (optional, not used by default)
    """
    import matplotlib.ticker as mticker
    x = tsne[:, 0]
    y = tsne[:, 1]
    X = np.column_stack([x, y])
    z = np.asarray(z)

    # 1. Grid
    x_grid = np.arange(x.min() - pad, x.max() + pad + grid_jump, grid_jump)
    y_grid = np.arange(y.min() - pad, y.max() + pad + grid_jump, grid_jump)
    xx, yy = np.meshgrid(x_grid, y_grid)
    grid = np.column_stack([xx.ravel(), yy.ravel()])

    # 2. Support mask
    nbrs = NearestNeighbors(n_neighbors=k).fit(X)
    d_train, _ = nbrs.kneighbors(X)
    train_kdist = d_train[:, -1]
    threshold = np.percentile(train_kdist, support_quantile)
    d_grid, _ = nbrs.kneighbors(grid)
    grid_kdist = d_grid[:, -1]
    mask = grid_kdist <= threshold

    # 3. Kernel regression
    Zhat = np.full(len(grid), np.nan)
    grid_in = grid[mask]
    dx = grid_in[:, None, 0] - X[None, :, 0]
    dy = grid_in[:, None, 1] - X[None, :, 1]
    dist2 = dx**2 + dy**2
    W = np.exp(-dist2 / (2 * bandwidth**2))
    Zhat[mask] = (W @ z) / (W.sum(axis=1) + 1e-12)
    Zgrid = Zhat.reshape(xx.shape)

    # 4. Plot contours with grid in background
    if axes is not None and ax_idx is not None:
        ax = axes[ax_idx]
    elif axes is not None and isinstance(axes, dict) and ax_idx is not None:
        ax = axes[ax_idx]
    else:
        fig, ax = plt.subplots(figsize=(3, 3))

    ax.xaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(mticker.MultipleLocator(1))
    ax.grid(True, which='major', linestyle='--', linewidth=0.5, alpha=0.3)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    # for spine in ax.spines.values():
    #     spine.set_visible(False)

    handles = []
    labels = []

    if fill:
        if contour_label:
            cf = ax.contourf(xx, yy, Zgrid, levels=levels, alpha=0.75, cmap="OrRd", label=contour_label)
        else:
            cf = ax.contourf(xx, yy, Zgrid, levels=levels, alpha=0.75, cmap="OrRd")
        # Add a colorbar as the color legend for the contour only if legend flag is passed
        if legend:
            cbar = plt.colorbar(cf, ax=ax, shrink=0.75, aspect=20, pad=0.03)
            if contour_label:
                cbar.set_label(contour_label, fontsize=8)
            else:
                cbar.set_label("Value", fontsize=8)
            cbar.ax.tick_params(labelsize=8)

    c = ax.contour(xx, yy, Zgrid, levels=levels, colors="black", linewidths=0.4, alpha=0.5)
    if text_on_contour:
        ax.clabel(c, inline=True, fontsize=8)

    # Scatter points (with legend if requested)
    scatter_artist = None
    if scatter and scatter_indices is not None:
        s_x = x[scatter_indices]
        s_y = y[scatter_indices]
        scatter_artist = ax.scatter(
            s_x, 
            s_y, 
            c="black", 
            s=40, 
            edgecolor="white", 
            zorder=10, 
            marker="X",
            label=scatter_label if legend else None
        )
        if legend:
            handles.append(scatter_artist)
            labels.append(scatter_label)

    # Arrow annotation (like notebook, but no legend or text)
    xlims = ax.get_xlim()
    ylims = ax.get_ylim()
    joint_x = xlims[0]
    joint_y = ylims[0]
    arr_len = 0.18 * (xlims[1] - xlims[0])
    if arrow_annotation:
        ax.annotate(
            "",
            xy=(joint_x + arr_len, joint_y),
            xytext=(joint_x, joint_y),
            arrowprops=dict(facecolor="black", width=0.1, headlength=5, headwidth=5),
            annotation_clip=False,
            zorder=10,
        )
        ax.annotate(
            "",
            xy=(joint_x, joint_y + arr_len),
            xytext=(joint_x, joint_y),
            arrowprops=dict(facecolor="black", width=0.1, headlength=5, headwidth=5),
            annotation_clip=False,
            zorder=10,
        )

    # Add the legend if requested and if any handle present:
    if legend and handles:
        ax.legend(handles=handles, labels=labels, loc="upper left", fontsize=8, frameon=True)

# ----------- MAIN LOGIC ------------
do_illu_best_worst = False
dataset_to_use = "gcn4"
settings = DATASET_SETTINGS[dataset_to_use]
contour_tsne_kwargs = settings["contour_tsne_kwargs"]
tsne_kwargs = settings["tsne_kwargs"]
EMBED = settings["embed"]
SAMPLE = settings["sample"]
N_SAMPLES = settings.get("n_samples", 15000)
K = settings.get("K", 100)
best_indices_list = settings["best"]  # per-dataset columns
worst_indices_list = settings["worst"]
pssm_indices_list = settings["pssm"]
esm8m_indices_list = settings["esm8m"]

output_dir = "/home/labs/fleishman/itayta/new_fitness_repo/fitness_learning/notebooks/data/tsne_embeddings"
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, f"tsne_embeddings_{dataset_to_use}.npy")

df = pd.read_csv(DATASET_PATHS[dataset_to_use])

if dataset_to_use == "gfp":
    activity = (~df["inactive"].to_numpy()).astype(int)
else:
    activity = df["activity"].to_numpy()
    gof_threshold = np.percentile(activity, 90)#1.2937578635
    activity /= gof_threshold
    #activity = (activity - np.nanmin(activity)) / (np.nanmax(activity) - np.nanmin(activity))

# get one-hot encoding for relevant columns
relevant_columns = get_relevant_columns(dataset_to_use, df)
ohe = get_one_hot_encoding(df, relevant_columns)
ohe_columns = ohe.columns
ohe = ohe.to_numpy().astype(int)

# Select indices:
gof_variants = np.where(activity > np.percentile(activity, 90))[0]

if SAMPLE:
    n_samples = min(N_SAMPLES, len(df))
    random_indices = np.random.choice(df.index, size=n_samples, replace=False)
    merged_indices = np.unique(np.concatenate([random_indices, gof_variants]))
else:
    merged_indices = np.arange(len(df))

# Slice activity for merged_indices for later plots
activity_selected = activity[merged_indices]
ohe_selected = ohe[merged_indices]

# t-SNE
if EMBED:
    embed_tsne(dataset_to_use, merged_indices, tsne_kwargs, output_path)

loaded_tsne_embs = np.load(output_path)
print("Loaded TSNE embeddings shape:", loaded_tsne_embs.shape)

# Calculate the actual top-K indices for each special set
def get_sorted_indices(column_idxs):
    # column_idxs: list of ohe columns for the motif/method
    return np.argsort(-ohe_selected[:, np.array(column_idxs)].sum(axis=1))


if do_illu_best_worst:
    indices_best = get_sorted_indices(best_indices_list)
    indices_worst = get_sorted_indices(worst_indices_list)
    indices_pssm = get_sorted_indices(pssm_indices_list)
    indices_esm8m = get_sorted_indices(esm8m_indices_list)

    # For each, get only top-K indices in merged_indices space
    K = min(K, loaded_tsne_embs.shape[0], len(indices_best))  # avoid overruns

    special_indices = [
        indices_best[:K],
        indices_worst[:K],
        indices_pssm[:K],
        indices_esm8m[:K],
    ]

    # Make a 1x4 plot, reusing axes like the notebook, no titles, no legends, proper export
    fig, axes = plt.subplots(1, 4, figsize=(8, 2))

    # Labels for the different sets to use for the scatter legend when desired
    scatter_labels = ["Best", "Worst", "PSSM", "ESM-8M"]

    # Add debug prints as in the notebook at relevant points
    for ax_idx, special_idxs in enumerate(special_indices):
        if ax_idx == 0:
            label = "best"
            special_columns = best_indices_list
        elif ax_idx == 1:
            label = "worst"
            special_columns = worst_indices_list
        elif ax_idx == 2:
            label = "pssm"
            special_columns = pssm_indices_list
        elif ax_idx == 3:
            label = "esm8m"
            special_columns = esm8m_indices_list
        else:
            label = "unknown"
            special_columns = []

        print([a.split("_")[0][1:] for a in ohe_columns[special_columns]])
        gof = (activity_selected > np.percentile(activity_selected, 90))
        # These prints follow the order in the notebook tsne_embeddings.ipynb snippet given
        print(f"Best gof sum:" if label == 'best' else
            f"Worst gof sum:" if label == 'worst' else
            f"PSSM gof sum:" if label == 'pssm' else
            f"Esm8m gof sum:", sum(gof[special_idxs]))
        print(f"Best gof sum:" if label == 'best' else
            f"Worst gof sum:" if label == 'worst' else
            f"PSSM gof sum:" if label == 'pssm' else
            f"Esm8m gof sum:", np.median(activity_selected[special_idxs]))

        # Enable legend only for the first panel or as needed
        want_legend = True if ax_idx == 0 else False
        do_legend = False
        contour_tsne_values(
            loaded_tsne_embs,
            activity_selected,
            scatter_indices=special_idxs,
            axes=axes,
            ax_idx=ax_idx,
            legend=do_legend,
            scatter_label=scatter_labels[ax_idx],
            **contour_tsne_kwargs,
        )

    plt.tight_layout()
    plt.savefig(
        f"./data/tsne_embeddings/tsne_contour_export_{dataset_to_use}_best_worst_new{'_with_legend' if do_legend else ''}.svg", format="svg"
    )
    plt.show()
else:
    esm_8m_indices = {
        # 3: [13, 343, 729],
        # 4: [13, 343, 729, 350],
        # 5: [13, 343, 729, 350, 264],
        6: [13, 343, 729, 350, 264, 588],
        7: [13, 343, 729, 350, 264, 588, 523],
        8: [13, 343, 729, 350, 264, 588, 523, 574],
        9: [13, 343, 729, 350, 264, 588, 523, 574, 46],
        10: [13, 343, 729, 350, 264, 588, 523, 574, 46, 411]}


    K_values = [
        5,
        10,
        20,
        50,
        100
    ]

    print("Doing illustration of ESM-8M matrix")

    fig, axes = plt.subplots(len(esm_8m_indices), len(K_values), figsize=(8, 8))

    # Add debug prints as in the notebook at relevant points
    for idx, (s, v) in enumerate(esm_8m_indices.items()):
        for jdx, K in enumerate(K_values):

            indices_esm8m = get_sorted_indices(v)
            special_idxs = indices_esm8m[:K]



        # Enable legend only for the first panel or as needed
            do_legend = False
            contour_tsne_values(
                loaded_tsne_embs,
                activity_selected,
                scatter_indices=special_idxs,
                axes=axes,
                ax_idx=(idx, jdx),
                legend=do_legend,
                scatter_label=f"ESM-8M {s} {K}",
                **contour_tsne_kwargs,
                arrow_annotation=False,
            )

    plt.tight_layout()
    plt.savefig(
        f"./data/tsne_embeddings/esm8m_matrix_plot_{dataset_to_use}.svg", format="svg"
    )
    plt.show()