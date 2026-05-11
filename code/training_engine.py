import yaml
import argparse


from trainer import *

# Example config.yaml:
# ---
# root_path: "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning"
# dataset_path: "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/configuration/fixed_unique_gfp_sequence_dataset_full_seq.csv"
# save_path: "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/pretraining/triplet_loss_backbones/one_shot/"
# weights_path: "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/pretraining/triplet_loss_backbones/final_model.pt"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and evaluate EpiNNet or PLM models.")
    parser.add_argument("--config",type=str,default="config.yaml",help="Path to the YAML configuration file (default: config.yaml)")
    parser.add_argument("--root_path", type=str, help="Root path for the project")
    parser.add_argument("--dataset_path", type=str, help="Path to the dataset CSV file")
    parser.add_argument("--save_path", type=str, help="Path to save outputs")
    parser.add_argument("--weights_path", type=str, help="Path to model weights")
    parser.add_argument("--model_type", type=str, help="Type of model to use (e.g., 'plm', 'epinnet')")
    parser.add_argument("--train_type", type=str, help="Type of training to use (e.g., 'direct_mlp')")
    parser.add_argument("--nmuts_column", type=str, help="Column name for number of mutations")
    parser.add_argument("--sequence_column_name", type=str, help="Column name for sequence")
    parser.add_argument("--label_column_name", type=str, help="Column name for activity")
    parser.add_argument("--first_column_name", type=str, help="First column name for encoding")
    parser.add_argument("--last_column_name", type=str, help="Last column name for encoding")
    parser.add_argument("--plm_name", type=str, help="PLM model name")
    parser.add_argument("--ref_seq", type=str, help="Reference sequence")
    parser.add_argument("--train_indices", type=int, nargs='+', help="List of train indices")
    parser.add_argument("--test_indices", type=int, nargs='+', help="List of test indices")
    parser.add_argument("--pos_to_use", default=None, type=int, nargs='+', help="List of positions to use")
    parser.add_argument("--flat_embeddings", type=lambda x: (str(x).lower() == 'true'), help="Whether to use flat embeddings (True/False)")
    parser.add_argument("--load_weights", type=lambda x: (str(x).lower() == 'true'), help="Whether to load weights (True/False)")
    parser.add_argument("--train", type=lambda x: (str(x).lower() == 'true'), help="Whether to train (True/False)")
    parser.add_argument("--train_indices_rev", type=lambda x: (str(x).lower() == 'true'), help="Reverse train indices (True/False)")
    parser.add_argument("--test_indices_rev", type=lambda x: (str(x).lower() == 'true'), help="Reverse test indices (True/False)")
    parser.add_argument("--evaluate_train", type=lambda x: (str(x).lower() == 'true'), help="Evaluate on train set (True/False)")
    parser.add_argument("--evaluate_test", type=lambda x: (str(x).lower() == 'true'), help="Evaluate on test set (True/False)")
    parser.add_argument("--train_drop_tokens", type=lambda x: (str(x).lower() == 'true'), help="Whether to drop tokens during training (True/False)")
    parser.add_argument("--inference_drop_tokens", type=lambda x: (str(x).lower() == 'true'), help="Whether to drop tokens during inference (True/False)")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--batch_size", type=int, help="Batch size")
    parser.add_argument("--iterations", type=int, help="Number of iterations")

    # After loading config, overwrite config values with any provided CLI args

    args = parser.parse_args()
    config_path = args.config

    #config_path = "msa_backbone_config.yaml"
    # Load configuration from YAML file
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Overwrite config values with any provided CLI args (if not None), dynamically
    for key, val in vars(args).items():
        if key == "config":
            continue  # skip config path itself

        if val is not None:
            config[key] = val


    if "root_path" in config:
        for k in list(config.keys()):
            if k.endswith("_path") and k != "root_path" and config[k] is not None:
                # Only update if not already an absolute path
                if not os.path.isabs(str(config[k])):
                    config[k] = os.path.join(config["root_path"], str(config[k]))

    plm_init(config["root_path"])


if __name__ == "__main__":
    if config["model_type"] == "plm":
        if config["train_type"] == "msa_backbone":
            train_evaluate_msa_backbone()
        else:
            train_evaluate_plms()
    elif config["model_type"] == "epinnet":
        train_evaluate_epinnet()
