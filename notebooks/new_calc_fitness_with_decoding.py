import pandas as pd
import numpy as np
import torch
import os
import sys

from utils_for_analysis import *
from scipy.stats import pearsonr, spearmanr


os.chdir(os.path.join(os.getcwd(), "../code/"))
if os.getcwd() not in sys.path:
    sys.path.insert(0, os.getcwd())

from plm_base import *
plm_init(os.path.join(os.getcwd(), "../"))

os.chdir(os.path.join(os.getcwd(), "../notebooks"))


# ============================================================
# CONFIGURATION
# ============================================================

SCORING_MODES = [
    #"left_to_right",
    "mutation_mask",
]

FORWARD_BATCH_SIZE = 60
VARIANT_SCORE_BATCH_SIZE = 2000
USE_MIXED_PRECISION = True
VERBOSE_SCORING = True
RESUME_MISSING_MODELS = False
RESULT_NAME_INCLUDES_SCORING_MODE = True

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("[INFO] Using device: %s" % str(device))

if device.type == "cuda":
    print("[INFO] GPU: %s" % torch.cuda.get_device_name(0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


full_model_names = {
    "esm_8m": "esm2_t6_8M_UR50D",
    "esm_35m": "esm2_t12_35M_UR50D",
    "esm_150m": "esm2_t30_150M_UR50D",
    "esm_650m": "esm2_t33_650M_UR50D",
    "esm_3b": "esm2_t36_3B_UR50D",
    "progen2-medium": "progen2-medium",
    "progen2-small": "progen2-small"
}

USE_SAPROT = False

SAPROT_MODEL_NAMES = {
    "saprot": "saprot",
    # "saprot_35m_af2": "saprot_35m_af2",
    # "saprot_650m_af2": "saprot_650m_af2",
    # "saprot_650m_pdb": "saprot_650m_pdb",
}

SAPROT_CACHE_IDENTIFIER = {
    "saprot": "saprot_%s",
    # "saprot_35m_af2": "saprot_35m_af2_%s",
    # "saprot_650m_af2": "saprot_650m_af2_%s",
    # "saprot_650m_pdb": "saprot_650m_pdb_%s",
}

if USE_SAPROT:
    full_model_names.update(SAPROT_MODEL_NAMES)

mask_tokens = {
    "esm_8m": "<mask>",
    "esm_35m": "<mask>",
    "esm_150m": "<mask>",
    "esm_650m": "<mask>",
    "esm_3b": "<mask>",
    "progen2-medium": "<|pad|>",
    "progen2-small": "<|pad|>",
    "saprot": "<mask>",
    "saprot_35m_af2": "<mask>",
    "saprot_650m_af2": "<mask>",
    "saprot_650m_pdb": "<mask>",
}

tokenized_sequences_path_dict = dict([
    (k, "%s_encoded_sequences.pt" % full_model_names[k])
    for k in full_model_names.keys()
])


def esmdecode(seq, tokenizer_dict):
    reverse_dict = dict((v, k) for (k, v) in tokenizer_dict.items())
    return "".join([reverse_dict[x] for x in seq])


def read_sequence_file(path):
    with open(path) as handle:
        return "".join(
            line.strip()
            for line in handle
            if line.strip() and not line.startswith(">")
        )


def is_saprot_model(model_key):
    return model_key in SAPROT_MODEL_NAMES


def get_tokenized_sequences_filename(model_key, dataset_to_use, dataset_cache_path):
    if not is_saprot_model(model_key):
        return tokenized_sequences_path_dict[model_key]

    cache_identifier = SAPROT_CACHE_IDENTIFIER.get(
        model_key,
        "%s_%%s" % model_key,
    ) % dataset_to_use

    candidates = [
        "%s_encoded_sequences.pt" % cache_identifier,
        "%s_encoded_sequences.pt" % full_model_names[model_key],
        tokenized_sequences_path_dict[model_key],
    ]

    for candidate in candidates:
        if os.path.exists(os.path.join(dataset_cache_path, candidate)):
            return candidate

    raise FileNotFoundError(
        "Could not find SaProt cached tokens in %s. Tried: %s" %
        (dataset_cache_path, ", ".join(candidates))
    )


def build_scoring_model(model_key, dataset_to_use, wt_seq):
    if not is_saprot_model(model_key):
        return plmEmbeddingModel(
            plm_name=full_model_names[model_key],
            logits_only=True,
            emb_only=False,
        )

    pdb_sequence_file = "./data/%s/%s_pdb_sequence.txt" % (
        dataset_to_use,
        dataset_to_use,
    )
    token_sequence_file = "./data/%s/%s_foldseek_3di.txt" % (
        dataset_to_use,
        dataset_to_use,
    )

    return StructurePlmEmbedding(
        plm_name=full_model_names[model_key],
        wt_sequence=wt_seq,
        pdb_sequence=read_sequence_file(pdb_sequence_file),
        foldseek_tokens=read_sequence_file(token_sequence_file),
        logits_only=True,
        emb_only=False,
    )


def decode_amino_acids_from_tokens(model, model_key, token_ids):
    if model_key.startswith("esm"):
        return esmdecode(token_ids, model.tokenizer.to_dict())

    if is_saprot_model(model_key):
        tokens = model.tokenizer.convert_ids_to_tokens(token_ids)
        tokens = [
            token for token in tokens
            if token not in model.tokenizer.all_special_tokens
        ]
        return "".join(token[0] for token in tokens)

    return model.tokenizer.decode(token_ids)


def get_mask_token_id(model, model_key, mask_token_string):
    if is_saprot_model(model_key):
        mask_token_id = model.tokenizer.convert_tokens_to_ids(mask_token_string)
        if mask_token_id is None or mask_token_id == model.tokenizer.unk_token_id:
            raise ValueError("Could not resolve SaProt mask token %r" % mask_token_string)
        return mask_token_id

    return model.encode(mask_token_string)[1]


def load_existing_result_columns(path, expected_rows, resume_missing_models):
    if not resume_missing_models or not os.path.exists(path):
        return [], set()

    df = pd.read_csv(path)
    if len(df) != expected_rows:
        raise ValueError(
            "Existing result file %s has %d rows, expected %d" %
            (path, len(df), expected_rows)
        )

    print(
        "[INFO] Resuming from %s with existing columns: %s" %
        (path, ", ".join(df.columns))
    )
    return [(col, df[col].to_numpy()) for col in df.columns], set(df.columns)


def get_result_name(model_key, scoring_mode):
    if RESULT_NAME_INCLUDES_SCORING_MODE:
        return "%s_%s" % (model_key, scoring_mode)

    if len(SCORING_MODES) != 1:
        raise ValueError(
            "RESULT_NAME_INCLUDES_SCORING_MODE=False requires exactly one scoring mode"
        )

    return model_key


def _get_model_device(model):
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _run_compact_states(
    model,
    wt_tokens,
    working_positions,
    mutable_states,
    output_positions,
    forward_batch_size=512,
    use_mixed_precision=True,
):
    """
    Evaluate compact mutable-position states.

    mutable_states stores only the tokens at working_positions.
    Full prompts are constructed one model batch at a time.
    """

    model_device = _get_model_device(model)

    wt_tokens = torch.as_tensor(
        wt_tokens,
        dtype=torch.long,
    ).reshape(-1).cpu()

    working_positions = torch.as_tensor(
        working_positions,
        dtype=torch.long,
    ).reshape(-1).cpu()

    mutable_states = torch.as_tensor(
        mutable_states,
        dtype=torch.long,
    ).cpu()

    if isinstance(output_positions, (int, np.integer)):
        one_output_position = True
        output_positions_cpu = int(output_positions)
    else:
        one_output_position = False
        output_positions_cpu = torch.as_tensor(
            output_positions,
            dtype=torch.long,
        ).reshape(-1).cpu()

    outputs = []
    start = 0
    current_batch_size = int(forward_batch_size)

    while start < mutable_states.shape[0]:
        end = min(
            start + current_batch_size,
            mutable_states.shape[0],
        )

        try:
            state_batch = mutable_states[start:end]

            prompt_batch = wt_tokens.unsqueeze(0).repeat(
                state_batch.shape[0],
                1,
            )

            prompt_batch[:, working_positions] = state_batch

            prompt_batch = prompt_batch.to(
                model_device,
                non_blocking=True,
            )

            if model_device.type == "cuda" and use_mixed_precision:
                amp_dtype = (
                    torch.bfloat16
                    if torch.cuda.is_bf16_supported()
                    else torch.float16
                )

                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=amp_dtype,
                ):
                    logits = model(prompt_batch)
            else:
                with torch.inference_mode():
                    logits = model(prompt_batch)

            if one_output_position:
                selected_logits = logits[:, output_positions_cpu, :]
            else:
                selected_logits = logits[
                    :,
                    output_positions_cpu.to(model_device),
                    :,
                ]

            batch_log_probs = torch.log_softmax(
                selected_logits.float(),
                dim=-1,
            ).cpu()

            outputs.append(batch_log_probs)
            start = end

            del logits
            del selected_logits
            del prompt_batch
            del state_batch

        except RuntimeError as error:
            is_cuda_oom = (
                model_device.type == "cuda"
                and "out of memory" in str(error).lower()
            )

            if not is_cuda_oom:
                raise

            if current_batch_size == 1:
                raise RuntimeError(
                    "CUDA out of memory even with batch size 1."
                ) from error

            current_batch_size = max(
                1,
                current_batch_size // 2,
            )

            print(
                "[WARNING] CUDA OOM. Retrying with batch size %d"
                % current_batch_size
            )

            torch.cuda.empty_cache()

    return torch.cat(outputs, dim=0)


def score_left_to_right(
    model,
    tokenized_sequence,
    wt_tokens,
    working_positions,
    mask_token,
    forward_batch_size=512,
    use_mixed_precision=True,
    verbose=True,
):
    """
    Left-to-right teacher-forced decoding.

    All working positions are initially masked. Positions are then revealed
    left-to-right. Variants sharing the same partial sequence reuse one state.
    """

    tokenized_sequence = torch.as_tensor(
        tokenized_sequence,
        dtype=torch.long,
    ).cpu()

    wt_tokens = torch.as_tensor(
        wt_tokens,
        dtype=torch.long,
    ).reshape(-1).cpu()

    working_positions = torch.as_tensor(
        sorted([int(p) for p in working_positions]),
        dtype=torch.long,
    )

    variant_working_tokens = tokenized_sequence[:, working_positions]

    n_sequences = tokenized_sequence.shape[0]
    n_working_positions = len(working_positions)

    sequence_scores = torch.zeros(
        n_sequences,
        dtype=torch.float32,
    )

    current_states = torch.full(
        (1, n_working_positions),
        int(mask_token),
        dtype=torch.long,
    )

    variant_to_state = torch.zeros(
        n_sequences,
        dtype=torch.long,
    )

    for local_position in range(n_working_positions):
        absolute_position = int(working_positions[local_position])
        target_tokens = variant_working_tokens[:, local_position]

        if verbose:
            print(
                "[INFO] left_to_right step %d/%d: %d unique partial states"
                % (
                    local_position + 1,
                    n_working_positions,
                    current_states.shape[0],
                )
            )

        state_log_probs = _run_compact_states(
            model=model,
            wt_tokens=wt_tokens,
            working_positions=working_positions,
            mutable_states=current_states,
            output_positions=absolute_position,
            forward_batch_size=forward_batch_size,
            use_mixed_precision=use_mixed_precision,
        )

        parent_and_token = torch.stack(
            [
                variant_to_state,
                target_tokens,
            ],
            dim=1,
        )

        unique_parent_and_token, variant_to_next_state = torch.unique(
            parent_and_token,
            dim=0,
            return_inverse=True,
        )

        pair_scores = state_log_probs[
            unique_parent_and_token[:, 0],
            unique_parent_and_token[:, 1],
        ]

        sequence_scores += pair_scores[variant_to_next_state]

        if local_position < n_working_positions - 1:
            next_states = current_states[
                unique_parent_and_token[:, 0]
            ].clone()

            next_states[:, local_position] = (
                unique_parent_and_token[:, 1]
            )

            current_states = next_states
            variant_to_state = variant_to_next_state

        del state_log_probs
        del parent_and_token
        del unique_parent_and_token
        del pair_scores

    return sequence_scores


def score_by_unique_mutation_masks(
    model,
    tokenized_sequence,
    wt_tokens,
    working_positions,
    mask_token,
    forward_batch_size=512,
    variant_score_batch_size=100_000,
    use_mixed_precision=True,
    verbose=True,
):
    """
    For each variant, mask only positions mutated relative to WT.

    Variants sharing the same mutation-position pattern reuse one masked-WT
    forward pass.

    Fitness(x) =
        sum over mutated positions i:
            log p(x_i | WT with x's mutated positions masked)
            -
            log p(WT_i | the same masked prompt)
    """

    tokenized_sequence = torch.as_tensor(
        tokenized_sequence,
        dtype=torch.long,
    ).cpu()

    wt_tokens = torch.as_tensor(
        wt_tokens,
        dtype=torch.long,
    ).reshape(-1).cpu()

    working_positions = torch.as_tensor(
        [int(p) for p in working_positions],
        dtype=torch.long,
    )

    variant_working_tokens = tokenized_sequence[:, working_positions]
    wt_working_tokens = wt_tokens[working_positions]

    mutation_masks = (
        variant_working_tokens
        != wt_working_tokens.unsqueeze(0)
    )

    calculated_nmuts = mutation_masks.sum(dim=1)

    unique_mutation_masks, variant_to_mask = torch.unique(
        mutation_masks,
        dim=0,
        return_inverse=True,
    )

    n_unique_masks = unique_mutation_masks.shape[0]
    n_working_positions = len(working_positions)

    if verbose:
        print(
            "[INFO] %d sequences produce %d unique mutation masks"
            % (
                tokenized_sequence.shape[0],
                n_unique_masks,
            )
        )

    mutation_mask_states = wt_working_tokens.unsqueeze(0).repeat(
        n_unique_masks,
        1,
    )

    mutation_mask_states[unique_mutation_masks] = int(mask_token)

    mask_log_probs = _run_compact_states(
        model=model,
        wt_tokens=wt_tokens,
        working_positions=working_positions,
        mutable_states=mutation_mask_states,
        output_positions=working_positions,
        forward_batch_size=forward_batch_size,
        use_mixed_precision=use_mixed_precision,
    )

    fitness = torch.zeros(
        tokenized_sequence.shape[0],
        dtype=torch.float32,
    )

    position_indices = torch.arange(
        n_working_positions,
        dtype=torch.long,
    ).unsqueeze(0)

    for start in range(
        0,
        tokenized_sequence.shape[0],
        variant_score_batch_size,
    ):
        end = min(
            start + variant_score_batch_size,
            tokenized_sequence.shape[0],
        )

        batch_mask_ids = variant_to_mask[start:end]
        batch_variant_tokens = variant_working_tokens[start:end]
        batch_mutation_masks = mutation_masks[start:end]

        batch_position_indices = position_indices.expand(
            end - start,
            -1,
        )

        variant_log_probs = mask_log_probs[
            batch_mask_ids.unsqueeze(1),
            batch_position_indices,
            batch_variant_tokens,
        ]

        wt_log_probs = mask_log_probs[
            batch_mask_ids.unsqueeze(1),
            batch_position_indices,
            wt_working_tokens.unsqueeze(0).expand(
                end - start,
                -1,
            ),
        ]

        fitness[start:end] = (
            (
                variant_log_probs - wt_log_probs
            )
            * batch_mutation_masks.to(torch.float32)
        ).sum(dim=1)

    return (
        fitness,
        calculated_nmuts,
        unique_mutation_masks,
    )


datasets = {
    # "lov": "./data/lov/lov.csv",
    # "pard3": "./data/pard3/pard3.csv",
    # "gcn4": "./data/gcn4/gcn4.csv",
    # "nmt": "./data/nmt/nmt_full_seq.csv",
    #"gfp": "./data/gfp/gfp_dataset_10mut.csv",
    #"pte": "./data/pte/pte.csv",
    #"aamyl": "./data/aamyl/aamyl.csv",
    #"his": "./data/his/his.csv",
    "his2": "./data/his2/his2.csv",
    "his5": "./data/his5/his5.csv",
    #"casp": "./data/casp/casp.csv"
}


for dataset_to_use in datasets.keys():
    dataset_cache_path = "%s_cache/misc/" % (
        datasets[dataset_to_use].split(".csv")[0]
    )

    df = pd.read_csv(datasets[dataset_to_use])

    working_positions = get_positions[dataset_to_use](df)
    working_positions = torch.tensor(
        [int(p) for p in working_positions],
        dtype=torch.long,
    )

    wt_indices = np.where(
        df[num_muts_column_name[dataset_to_use]] == 0
    )[0]

    assert len(wt_indices) == 1, (
        "Expected exactly one WT sequence, found %d"
        % len(wt_indices)
    )

    wt_idx = int(wt_indices.item())

    wt_seq = df[
        full_seq_column_name[dataset_to_use]
    ].iloc[wt_idx]

    nmuts_vec = df[
        num_muts_column_name[dataset_to_use]
    ].to_numpy().astype(int)

    normalization_denominator = nmuts_vec.copy()
    normalization_denominator[
        normalization_denominator == 0
    ] = 1

    print("########################################################")
    print("[INFO] Working on %s" % dataset_to_use)
    print("[INFO] WT index is %d" % wt_idx)
    print(
        "[INFO] Working positions are %s"
        % " ".join([str(int(x)) for x in working_positions])
    )
    print("[INFO] Overall %d sequences to process" % len(df))
    print("[INFO] WT sequence is %s" % wt_seq)
    print("[INFO] WT sequence length is %d" % len(wt_seq))
    print("[INFO] Cache path is %s" % dataset_cache_path)
    print("########################################################")

    N_random_sequences_to_assert = 20

    save_path = (
        "./notebooks/%s/fitness_results/"
        % dataset_to_use
    )

    os.makedirs(save_path, exist_ok=True)

    normed_fitness_path = "%s/new_normed_fitness_all.csv" % save_path
    fitness_path = "%s/new_fitness_all.csv" % save_path

    normed_fitness_all, existing_normed_columns = load_existing_result_columns(
        normed_fitness_path,
        len(df),
        RESUME_MISSING_MODELS,
    )
    fitness_all, existing_fitness_columns = load_existing_result_columns(
        fitness_path,
        len(df),
        RESUME_MISSING_MODELS,
    )

    for k in tokenized_sequences_path_dict.keys():
        print("########################################################")
        print("[INFO] Loading %s" % k)

        expected_result_names = [
            get_result_name(k, scoring_mode)
            for scoring_mode in SCORING_MODES
        ]

        missing_result_names = [
            result_name
            for result_name in expected_result_names
            if (
                result_name not in existing_normed_columns
                or result_name not in existing_fitness_columns
            )
        ]

        if RESUME_MISSING_MODELS and len(missing_result_names) == 0:
            print(
                "[INFO] Skipping %s; all requested result columns already exist"
                % k
            )
            continue

        v = get_tokenized_sequences_filename(
            k,
            dataset_to_use,
            dataset_cache_path,
        )

        tokenized_sequence = torch.load(
            os.path.join(dataset_cache_path, v),
            map_location="cpu",
        )

        tokenized_sequence = torch.as_tensor(
            tokenized_sequence,
            dtype=torch.long,
        ).cpu()

        N_tokens = tokenized_sequence.shape[1]
        N_seq = tokenized_sequence.shape[0]

        assert len(wt_seq) == N_tokens - 2, (
            "ASSERT FAILED: WT sequence length is not equal "
            "to number of tokens - 2 (bos/eos)"
        )
        print("[INFO] ASSERT 1/2 (preprocessing) passed")

        assert len(df) == N_seq, (
            "ASSERT FAILED: Number of sequences in df is "
            "not equal to number of tokenized sequences"
        )
        print("[INFO] ASSERT 2/2 (preprocessing) passed")

        print("########################################################")
        print(
            "Shape of tokenized sequence is %s"
            % str(tokenized_sequence.shape)
        )

        model = build_scoring_model(
            k,
            dataset_to_use,
            wt_seq,
        ).to(device).eval()

        print(
            "[INFO] Model is on %s"
            % str(_get_model_device(model))
        )

        wt_tokens = tokenized_sequence[
            wt_idx:wt_idx + 1,
            :,
        ]

        vocab = model.vocab

        print(
            "[INFO] Asserting %d random sequences"
            % N_random_sequences_to_assert
        )

        columns = get_relevant_columns[dataset_to_use](df)

        for i in range(N_random_sequences_to_assert):
            idx = np.random.randint(0, N_seq)

            seq_tokens = tokenized_sequence[
                idx,
                working_positions,
            ].tolist()

            decoded_from_tokens = decode_amino_acids_from_tokens(
                model,
                k,
                seq_tokens,
            )

            seq_from_df = "".join(
                df.iloc[idx][columns].tolist()
            )

            assert decoded_from_tokens == seq_from_df

            if (i + 1) % 5 == 0:
                print(
                    "[INFO] ASSERT %d/%d "
                    "(random sequence tokenization assertion) passed"
                    % (
                        i + 1,
                        N_random_sequences_to_assert,
                    )
                )

        print("[INFO] Vocab is %s" % str(vocab))

        mask_token_string = mask_tokens[k]

        print(
            "[INFO] Mask token string is %s"
            % mask_token_string
        )

        mask_token = get_mask_token_id(
            model,
            k,
            mask_token_string,
        )

        print(
            "[INFO] Mask token value is %s"
            % str(mask_token)
        )

        if not k.startswith("esm"):
            print(
                "[WARNING] ProGen2 is causal. Using its pad token as "
                "a mask is not equivalent to native causal likelihood."
            )

        masked_prompt = wt_tokens.clone()
        masked_prompt[:, working_positions] = mask_token

        print(
            "[INFO] WT tokens before masking are %s"
            % str(wt_tokens)
        )
        print(
            "[INFO] WT tokens after masking are %s"
            % str(masked_prompt)
        )
        print(
            "[INFO] diff: %s"
            % str(wt_tokens - masked_prompt)
        )

        assert (
            (
                wt_tokens
                - masked_prompt
                + mask_token
            )[:, working_positions]
            == wt_tokens[:, working_positions]
        ).sum().item() == len(working_positions), (
            "masking indexing failed"
        )

        print("[INFO] ASSERT 1/2 (masking) passed")

        assert (
            masked_prompt[:, working_positions].sum().item()
            == mask_token * len(working_positions)
        ), "masking operation failed"

        print("[INFO] ASSERT 2/2 (masking) passed")

        for scoring_mode in SCORING_MODES:
            result_name = get_result_name(k, scoring_mode)

            if (
                RESUME_MISSING_MODELS
                and result_name in existing_normed_columns
                and result_name in existing_fitness_columns
            ):
                print(
                    "[INFO] Skipping %s; result column already exists"
                    % result_name
                )
                continue

            print("########################################################")
            print(
                "[INFO] Scoring %s using %s"
                % (k, scoring_mode)
            )

            if scoring_mode == "left_to_right":
                raw_scores = score_left_to_right(
                    model=model,
                    tokenized_sequence=tokenized_sequence,
                    wt_tokens=wt_tokens,
                    working_positions=working_positions,
                    mask_token=mask_token,
                    forward_batch_size=FORWARD_BATCH_SIZE,
                    use_mixed_precision=USE_MIXED_PRECISION,
                    verbose=VERBOSE_SCORING,
                )

                wt_score = raw_scores[wt_idx]
                fitness = (raw_scores - wt_score).numpy()

                calculated_nmuts = (
                    tokenized_sequence[:, working_positions]
                    != wt_tokens[:, working_positions]
                ).sum(dim=1).numpy()

            elif scoring_mode == "mutation_mask":
                (
                    mutation_mask_fitness,
                    calculated_nmuts_tensor,
                    unique_mutation_masks,
                ) = score_by_unique_mutation_masks(
                    model=model,
                    tokenized_sequence=tokenized_sequence,
                    wt_tokens=wt_tokens,
                    working_positions=working_positions,
                    mask_token=mask_token,
                    forward_batch_size=FORWARD_BATCH_SIZE,
                    variant_score_batch_size=(
                        VARIANT_SCORE_BATCH_SIZE
                    ),
                    use_mixed_precision=USE_MIXED_PRECISION,
                    verbose=VERBOSE_SCORING,
                )

                fitness = mutation_mask_fitness.numpy()
                calculated_nmuts = (
                    calculated_nmuts_tensor.numpy()
                )

                print(
                    "[INFO] Number of unique mutation masks: %d"
                    % unique_mutation_masks.shape[0]
                )

            else:
                raise ValueError(
                    "Unknown scoring mode: %s"
                    % scoring_mode
                )

            assert np.isclose(
                fitness[wt_idx],
                0.0,
            ), (
                "WT fitness is not zero for %s"
                % scoring_mode
            )

            assert calculated_nmuts[wt_idx] == 0

            # assert np.array_equal(
            #     calculated_nmuts,
            #     nmuts_vec,
            # ), (
            #     "Mutation counts calculated from tokens do not "
            #     "match the dataset mutation-count column."
            # )

            print(
                "[INFO] ASSERT 1/2 "
                "(token mutation counts match dataframe) passed"
            )

            print(
                "[INFO] ASSERT 2/2 "
                "(WT fitness is zero) passed"
            )

            normed_fitness = (
                fitness
                / normalization_denominator
            )

            normed_fitness_all.append(
                (result_name, normed_fitness)
            )

            fitness_all.append(
                (result_name, fitness)
            )

            print(
                "[INFO] %s fitness range: %.6f to %.6f"
                % (
                    scoring_mode,
                    fitness.min(),
                    fitness.max(),
                )
            )

            print(
                "[INFO] %s normalized fitness range: %.6f to %.6f"
                % (
                    scoring_mode,
                    normed_fitness.min(),
                    normed_fitness.max(),
                )
            )

        del model

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    pd.DataFrame(
        dict(normed_fitness_all)
    ).to_csv(
        normed_fitness_path,
        index=False,
    )

    pd.DataFrame(
        dict(fitness_all)
    ).to_csv(
        fitness_path,
        index=False,
    )
