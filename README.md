# PLM Benchmark Analyses

This repository contains the notebooks and scripts used to benchmark protein language models (pLMs), simple baselines, and controlled extrapolation settings in mutation-dense protein design tasks.

## Installation and data download

To reproduce the analyses, first clone the repository:

```bash
git clone https://github.com/drprfitay/fitness_learning.git
cd fitness_learning
```

Create and switch to the submission branch:

```bash
git checkout Submission_V1
```

Then run the environment installer:

```bash
chmod +x installer.sh
./installer.sh
```

After installation, activate the conda environment if it is not already active:

```bash
conda activate esm_env
```

Then download and extract the required data files:

```bash
chmod +x downloader.sh
./downloader.sh
```

After these steps, the notebooks and analysis scripts can be run from the repository.


## Main analysis notebooks

### `plm_zero_shot.ipynb`

Zero-shot analysis comparing pLM scores against the PSSM baseline.

### `analysis_of_variance.ipynb`

Analysis of variance, controlled extrapolation analysis, and comparisons between different pLMs.

### `classifier_over_embeddings.ipynb`

Controlled extrapolation using classifiers trained over pLM embeddings, including comparison to one-hot encoding in controlled extrapolation settings.

### `by_subsamples_all_new.ipynb`

Random train/test split analysis comparing pLM embeddings against one-hot encoding across different training sample sizes.

### `rank_mutations.ipynb`

Library simulation and generation-like benchmarking, including mutation ranking and model-guided selection analyses.

## Example commands

### Train classifier over pLM embeddings

Controlled extrapolation example:

```bash
python train_classifiers_over_embeddings.py \
    --model_name esm_8m \
    --dataset_name pard3 \
    --mean_embeddings \
    --regression \
    --external_labels_column activity \
    --n_start 1 \
    --n_end 6
```

### Train classifier over one-hot encoding

Controlled extrapolation example:

```bash
python train_classifiers_over_ohe.py \
    --dataset pard3 \
    --external_labels_column activity \
    --regression \
    --n_start 1 \
    --n_end 6
```

### Calculate pLM zero-shot scores

```bash
python calculate_fitness_for_plm.py
```

Note: this step may take a while on new machines, especially when running large protein language models.

### Train regressors on random splits

```bash
python train_regressors_over_embeddings_subsamples.py \
    --dataset_name pard3 \
    --n_samples 100 200 500 1000 5000
```

## Notes

- The controlled extrapolation analyses train on variants up to a specified mutational order and evaluate on higher-order variants.
- The random split analyses evaluate model performance under conventional train/test splitting.
- Zero-shot analyses compare pretrained pLM scores against evolutionary baselines such as PSSM.
- Library simulation analyses mimic model-guided protein design by asking whether model-selected mutational neighborhoods are enriched for active or gain-of-function variants.
