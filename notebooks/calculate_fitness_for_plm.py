
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

full_model_names = {
    "esm_8m": "esm2_t6_8M_UR50D",
    "esm_35m": "esm2_t12_35M_UR50D",
    "esm_150m": "esm2_t30_150M_UR50D",
    "esm_650m": "esm2_t33_650M_UR50D",
    "esm_3b": "esm2_t36_3B_UR50D",
    "progen2-medium": "progen2-medium",
    "progen2-small": "progen2-small"
    #"prot_bert": "prot_bert" # bert has no decoder
}

mask_tokens = {
    "esm_8m": "<mask>",
    "esm_35m": "<mask>",
    "esm_150m": "<mask>",
    "esm_650m": "<mask>",
    "esm_3b": "<mask>",
    "progen2-medium": "<|pad|>",
    "progen2-small": "<|pad|>"
    #"prot_bert": "[MASK]" 
}

tokenized_sequences_path_dict = dict([k, "%s_encoded_sequences.pt" % full_model_names[k]] for k in full_model_names.keys())

def esmdecode(seq, tokenizer_dict):
    reverse_dict = dict((v,k) for (k,v) in tokenizer_dict.items())
    return "".join([reverse_dict[x] for x in seq])

datasets = {
    #"lov": "./data/lov/lov.csv",
    #"pard3": "./data/pard3/pard3.csv",
    #"gcn4": "./data/gcn4/gcn4.csv",
    #"nmt": "./data/nmt/nmt_full_seq.csv",
    #"gfp" : "./data/gfp/gfp_dataset_10mut.csv",
    #"pte": "./data/pte/pte.csv" 
    #"aamyl": "./data/aamyl/aamyl.csv",
    #"his": "./data/his/his.csv",
    "casp": "./data/casp/casp.csv"
}



for dataset_to_use in datasets.keys():
    dataset_cache_path = "%s_cache/misc/" % datasets[dataset_to_use].split(".csv")[0]
    df = pd.read_csv(datasets[dataset_to_use])
    working_positions = get_positions[dataset_to_use](df)
    working_positions = torch.tensor([int(p) for p in working_positions])
    wt_idx = np.where(df[num_muts_column_name[dataset_to_use]] == 0)[0]
    wt_seq = df[full_seq_column_name[dataset_to_use]].iloc[wt_idx].item()
    nmuts_vec = df[num_muts_column_name[dataset_to_use]].to_numpy()
    nmuts_vec[wt_idx] = 1

    print("########################################################")
    print("[INFO] Working on %s" % dataset_to_use)
    print("[INFO] WT index is %d" % wt_idx)
    print("[INFO] Working positions are %s" %  " ".join([str(x) for x in working_positions]))
    print("[INFO] Overall %d sequences to process" % len(df))
    print("[INFO] WT sequence is %s" % wt_seq)
    print("[INFO] WT sequence length is %d" % len(wt_seq))
    print("[INFO] Cahce path is %s" % dataset_cache_path)
    print("########################################################")

    # ASSERT  model.tokenizer.decode(tokenized_sequence[:,working_positions][0].tolist()) == "".join(df.iloc[0][df.columns[si:ei][positions_with_mutations]].tolist())

    normed_fitness_all = []
    fitness_all = []
    N_random_sequences_to_assert = 20
    save_path = "./notebooks/%s/fitness_results/" % dataset_to_use
    os.makedirs(save_path, exist_ok=True)
    for k,v in tokenized_sequences_path_dict.items():
        print("########################################################")
        print("[INFO] Loading %s" % k)

        tokenized_sequence = torch.load(os.path.join(dataset_cache_path, v))
        N_tokens = tokenized_sequence.shape[1]
        N_seq = tokenized_sequence.shape[0]

        assert len(wt_seq) == N_tokens - 2, "ASSERT FAILED: WT sequence length is not equal to number of tokens - 2 (bos/eos)"
        print("[INFO] ASSERT 1/2 (preprocessing) passed")

        assert len(df) == N_seq, "ASSERT FAILED: Number of sequences in df is not equal to number of tokenized sequences"
        print("[INFO] ASSERT 2/2 (preprocessing) passed")

        print("########################################################")
        print("Shape of tokenized sequence is %s" % str(tokenized_sequence.shape))    
        model = plmEmbeddingModel(plm_name=full_model_names[k], logits_only=True, emb_only=False).eval()
        wt_tokens = tokenized_sequence[wt_idx, :]
        vocab = model.vocab


        print("[INFO] Asserting %d random sequences" % N_random_sequences_to_assert)
        for i in range(N_random_sequences_to_assert):            
            idx = np.random.randint(0, N_seq)        
            
            seq_tokens = tokenized_sequence[:,working_positions][idx].tolist()
            
            if k.startswith("esm"):
                decoded_from_tokens = esmdecode(seq_tokens, model.tokenizer.to_dict())
            else:
                decoded_from_tokens = model.tokenizer.decode(seq_tokens)
            columns = get_relevant_columns[dataset_to_use](df)
            seq_from_df = "".join(df.iloc[idx][columns].tolist())       

            assert  decoded_from_tokens == seq_from_df       
            if (i + 1) % 5 == 0:
                print("[INFO] ASSERT %d/%d (random sequence tokenization assertion) passed" % (i+1, N_random_sequences_to_assert))


        print("[INFO] Vocab is %s" % str(vocab))
        mask_token_string = mask_tokens[k]

        print("[INFO] Mask token string is %s" % mask_token_string)
        mask_token = model.encode(mask_token_string)[1]

        print("[INFO] Mask token value is %s" % str(mask_token))
        print("[INFO] WT tokens before masking are %s, setting mask" % str(wt_tokens))

        masked_prompt = wt_tokens.clone()
        # PDB INDEX + PAD so if the working position is 1, it should be "0", but because pad it's 1
        masked_prompt[:, working_positions] = mask_token 

        print("[INFO] WT tokens after masking are %s" % str(wt_tokens))
        print("[INFO] diff: %s" % str(wt_tokens - masked_prompt))


        assert ((wt_tokens - masked_prompt + mask_token)[:,working_positions] == wt_tokens[:, working_positions]).sum().item() == len(working_positions), "masking indexing failed"
        print("[INFO] ASSERT 1/2 (masking) passed")

        assert masked_prompt[:,working_positions].sum().item() == mask_token * len(working_positions), "masking operation failed"
        print("[INFO] ASSERT 2/2 (masking) passed")

        logits = model(masked_prompt)
        pssm = logits.softmax(dim=2)
        pssm = pssm.squeeze(0)

        assert pssm.sum(dim=1).sum().to(torch.int).item() == N_tokens
        print("[INFO] ASSERT 1/1 (logits forward) passed")

        wt_one_hot = torch.nn.functional.one_hot(wt_tokens.squeeze(0), pssm.shape[1])
        variant_one_hot = torch.nn.functional.one_hot(torch.tensor(tokenized_sequence), pssm.shape[1])
        probs_per_pos_variant = torch.einsum("BSV,SV->BS", variant_one_hot.to(torch.float), pssm)[:,working_positions]
        probs_per_pos_wt = (pssm * wt_one_hot).sum(dim=1)[working_positions]

        print((probs_per_pos_variant - probs_per_pos_wt))
        # -1 due to WT

        
        
        #assert (((probs_per_pos_variant - probs_per_pos_wt) != 0).sum(dim=1).numpy() == nmuts_vec).sum() == (N_seq - 1), "fitness calculation failed"
        print("[INFO] ASSERT 1/1 (fitness calculation) passed")

        fitness = (torch.log(probs_per_pos_variant) - torch.log(probs_per_pos_wt)).sum(dim=1).detach().numpy()
        
        print(np.unique(nmuts_vec, return_counts=True))
        normed_fitness_all.append((k, fitness / nmuts_vec))
        fitness_all.append((k, fitness))

        print((fitness / nmuts_vec) - fitness)

        print("###")
        print(pssm.shape)
        print(pssm[working_positions].shape)
        
        # Set the column names of the pssm DataFrame to vocab
        #pd.DataFrame(pssm[working_positions].detach().numpy(), columns=vocab).to_csv("%s/pssm_%s.csv" % (save_path, k), index=False)
    pd.DataFrame(dict(normed_fitness_all)).to_csv("%s/normed_fitness_all.csv" % save_path, index=False)
    pd.DataFrame(dict(fitness_all)).to_csv("%s/fitness_all.csv" % save_path, index=False)




