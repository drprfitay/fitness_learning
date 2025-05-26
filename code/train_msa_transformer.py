# -*- coding: utf-8 -*-

import sys, os
import torch
import torch.nn.functional as F
import loralib as lora
import scipy.stats
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import einops

from esm_smart_dataset import *
from sequence_space_utils import *

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Align import substitution_matrices
from Bio import SeqIO

from random import sample

blosum62 = substitution_matrices.load("BLOSUM62")


ROOT_PATH = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning"
RESULTS_PATH = "%s/results" % ROOT_PATH
ITAYFOLD_PATH = "%s/itayFold/" % ROOT_PATH
WEIGHTS_PATH = "/%s/weights/" % ITAYFOLD_PATH

MSA_PATH = "%s/data/datasets/msa/" % ROOT_PATH

STOCK_MSAS = ["1a3a_1_A.a3m", 
              "5ahw_1_A.a3m", 
              "1xcr_1_A.asm",
              "gfp2.aln"]

MODEL_WEIGHTS_FILE_NAME = "esm3/esm_model_weights.pth"
LORA_WEIGHTS_FIlE_NAME =  "esm3/esm_lora_weights.pth"
ENCODER_WEIGHTS_FILE_NAME = "esm3/structure_encoder.pth"
DECODER_WEIGHTS_FILE_NAME = "esm3/structure_decoder.pth"



ROOT_DMS_PATH = "%s/data/datasets/DMS/Data" % ROOT_PATH
BASE_DMS_PATH = "%s/data/" % ROOT_DMS_PATH
BASE_DMS_PDB_PATH = "%s/structure_data/" % ROOT_DMS_PATH 

def fix_esm_path():
    global original_sys_path
    
    # Specify the module name and path
    module_name = "esm"
    module_path = ITAYFOLD_PATH 
    
    # Store the original sys.path
    original_sys_path = sys.path.copy()

    # Temporarily add the local directory to sys.path
    sys.path.insert(0, os.path.abspath(module_path))

    # hack
    for mdl in [k for k,v in sys.modules.items() if module_name in k]:
        del sys.modules[mdl]

fix_esm_path()

import esm2
import string


deletekeys = dict.fromkeys(string.ascii_lowercase)
deletekeys["."] = None
deletekeys["*"] = None
translation = str.maketrans(deletekeys)
    
def read_sequence(filename: str) -> tuple[str, str]:
    """ Reads the first (reference) sequences from a fasta or MSA file."""
    record = next(SeqIO.parse(filename, "fasta"))
    return record.description, str(record.seq)

def remove_insertions(sequence: str) -> str:
    """ Removes any insertions into the sequence. Needed to load aligned sequences in an MSA. """
    return sequence.translate(translation)

def read_msa(filename: str) -> list[tuple[str, str]]:
    """ Reads the sequences from an MSA file, automatically removes insertions."""
    return [(record.description, remove_insertions(str(record.seq))) for record in SeqIO.parse(filename, "fasta")]


def load_esm2_model_and_alphabet(model_name):
    supported_esm2_models =\
        ["esm1_t34_670M_UR50S",
         "esm1_t34_670M_UR50D",
         "esm1_t34_670M_UR100",
         "esm1_t12_85M_UR50S",
         "esm1_t6_43M_UR50S",
         "esm1b_t33_650M_UR50S",
         "esm_msa1_t12_100M_UR50S",
         "esm_msa1b_t12_100M_UR50S",        
         "esm1v_t33_650M_UR90S_1",
         "esm1v_t33_650M_UR90S_2",
         "esm1v_t33_650M_UR90S_3",
         "esm1v_t33_650M_UR90S_4",
         "esm1v_t33_650M_UR90S_5",
         "esm_if1_gvp4_t16_142M_UR50",
         "esm2_t6_8M_UR50D",
         "esm2_t12_35M_UR50D",
         "esm2_t30_150M_UR50D",
         "esm2_t33_650M_UR50D",
         "esm2_t36_3B_UR50D",
         "esm2_t48_15B_UR50D"]
        
    if model_name not in supported_esm2_models:
        print("Unsupported model %s" % model_name)
        
    model_weights_and_data_path = "%s/esm2/%s.pth" % (WEIGHTS_PATH, model_name)
    
    if model_weights_and_data_path in os.listdir("%s/esm2" % WEIGHTS_PATH):
        model_data = torch.load(model_weights_and_data_path)
    else:    
        model_data, regression_data = esm2.pretrained._download_model_and_regression_data(model_name)
        
        if regression_data is not None:
            model_data["model"].update(regression_data["model"])
            
        # Save model data
        torch.save(model_data, model_weights_and_data_path)
        
    return esm2.pretrained.load_model_and_alphabet_core(model_name, 
                                                        model_data, 
                                                        regression_data=None)
        

def get_indices(sequence_df, nmuts, rev=False, verbose=False):
        
    indices = np.repeat(False, sequence_df.shape[0])
        
    if type(nmuts) == int:
        nmuts = [nmuts]
            
    for nm in nmuts:
        indices = indices | (sequence_df["num_of_muts"] == nm).to_numpy()
        if verbose:
            print("Indices included: %d" % sum(indices))
            
    if rev:
        indices = ~indices
            
    return(np.where(indices)[0].tolist())
        

def msat_baseline():
    model_name = "esm_msa1b_t12_100M_UR50S"
    esm2_model,esm2_alphabet  = load_esm2_model_and_alphabet(model_name)
        
        
    base_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/configuration/"
    #dataset_path = "%s/fixed_unique_gfp_sequence_dataset.csv" % base_path
    dataset_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq.csv" % base_path    
    output_dir = "%s/gfp_dataset/esm2" % RESULTS_PATH
    os.makedirs(output_dir, exist_ok=True)
        
    
    
        # This is bad, I had to fish it from this link: https://github.com/Fleishman-Lab/htFuncLib/blob/main/fluroescene_vs_the_number_of_mutations/GFP_threshold_epistasis.ipynb
    jonathans_reference_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"
        
    dataset = Esm2SequenceActivityDataset(dataset_path,
                                          esm2_alphabet,
                                          get_mutated_position_function=get_mutated_position_function_gfp,
                                          cache=True,
                                          model_name="msa_transformer",
                                          activity_column_name='is_unsorted',
                                          sequence_column_name="FullSeq",
                                          ref_seq=jonathans_reference_sequence,
                                          labels_dtype=torch.int64)
    
    masked_tensor = torch.unique(dataset.one_hot_mut_info[2], dim=0)
    pad=torch.tensor(0).view((1,-1))
    padded_masked_tensor = torch.cat([pad, masked_tensor, pad], dim=1)
        
    wt_tokens = torch.tensor(esm2_alphabet.encode("<cls>" + jonathans_reference_sequence + "<eos>"), dtype=torch.int64).view((1,-1))
        
    sequence_mask_token = torch.tensor(esm2_alphabet.encode("<mask>"), dtype=torch.int64)
    batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * wt_tokens)
    batched_sequences_to_run_with_masks += padded_masked_tensor * sequence_mask_token # add masks                
    
    

    
    msa = read_msa("%s/%s" % (MSA_PATH, STOCK_MSAS[3]))
    sampled = sample(range(1,1019), 150) # ignore 0 on purpose
    msa = [m for i,m in enumerate(msa) if i in sampled]
    msa_converter = esm2_alphabet.get_batch_converter()
    #     logits = esm2_model.forward(batched_sequences_to_run_with_masks)
    labels, batch_strs, msa_tokens = msa_converter(msa)
    eos_token = torch.tensor(esm2_alphabet.encode("<eos>"), dtype=torch.int64)
    torch.cat([msa_tokens, torch.ones((1, msa_tokens.shape[1], 1)) * eos_token], dim=2)
    msa_tokens = torch.cat([msa_tokens, torch.ones((1, msa_tokens.shape[1], 1)) * eos_token], dim=2)
    msa_tokens_with_mask = torch.cat([batched_sequences_to_run_with_masks.view((1,1,-1)), msa_tokens[:,1:,]], dim=1)
    
    logits = esm2_model.forward(msa_tokens_with_mask.to(torch.int64))
    masked_logits = logits["logits"][:,0,:,:]
    pssm = masked_logits.softmax(dim=2)
        
    from_mutation = dataset.one_hot_mut_info[0]
    to_mutation = dataset.one_hot_mut_info[1]
    padded_mutated_positions = padded_masked_tensor == 1
    mutated_positions = masked_tensor == 1
        
    sns.heatmap(pssm[0][:,4:29].detach().numpy())
    sns.heatmap(pssm[padded_mutated_positions,:][:,4:29].detach().numpy())
        
    fixed_pssm = pssm[:,1:-1,:].view((len(jonathans_reference_sequence), -1))
        
    predicted_fitness =\
        fitness_from_prob_non_dms(fixed_pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
        from_mutation[:,mutated_positions.view((-1))][0],
        to_mutation[:,mutated_positions.view((-1))])
            
    res_df = pd.DataFrame(np.transpose(predicted_fitness.detach().numpy()))
    res_df.to_csv("%s/%s_predicted_fitness.csv" % (output_dir, model_name))
        
    
def train_msa_transformer():
    
    dpo_beta = .5
    BATCH_SIZE = 256
    root_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/"
    base_path = "%s/data/configuration/" % root_path
    train_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq_train.csv" % base_path
    saved_weights_path = "/%s/itayFold/weights/esm2/retrained/msa_transformer" % root_path
    
    model_name = "esm_msa1b_t12_100M_UR50S"
    trainable_msa_transformer, esm2_alphabet  = load_esm2_model_and_alphabet(model_name)
    ref_msa_transformer, esm2_alphabet  = load_esm2_model_and_alphabet(model_name)
    msa_converter = esm2_alphabet.get_batch_converter()     
    
    for p in ref_msa_transformer.parameters(): p.requires_grad = False
    
    jonathans_reference_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"


    dataset2 = \
        Esm2SequenceActivityContrastiveDatasetAdvancedMask(train_path,
                                               esm2_alphabet,
                                               get_mutated_position_function=get_mutated_position_function_gfp_n2,
                                               cache=True,
                                               model_name="msa_transformer",
                                               activity_column_name='is_unsorted',
                                               sequence_column_name="FullSeq",
                                               ref_seq=jonathans_reference_sequence,
                                               labels_dtype=torch.int64)

    dataset = \
        Esm2SequenceActivityContrastiveDataset(train_path,
                                               esm2_alphabet,
                                               get_mutated_position_function=get_mutated_position_function_gfp,
                                               cache=True,
                                               model_name="msa_transformer",
                                               activity_column_name='is_unsorted',
                                               sequence_column_name="FullSeq",
                                               ref_seq=jonathans_reference_sequence,
                                               labels_dtype=torch.int64)
    
    
    #masked_tensor = torch.unique(dataset.one_hot_mut_info[2], dim=0)
    masked_tensor = dataset.wt_one_hot#torch.unique(dataset.wt_one_hot, dim=0)
    pad=torch.tensor(0).view((1,-1))
    padded_masked_tensor = torch.cat([pad, masked_tensor.view((1,-1)), pad], dim=1)
        
    padded_mutated_positions = (padded_masked_tensor == 1).view(-1)
    mutated_positions = masked_tensor == 1
    
    wt_tokens = torch.tensor(esm2_alphabet.encode("<cls>" + jonathans_reference_sequence + "<eos>"), dtype=torch.int64).view((1,-1))
    eos_token = torch.tensor(esm2_alphabet.encode("<eos>"), dtype=torch.int64)    
    mask_token = torch.tensor(esm2_alphabet.encode("<mask>"), dtype=torch.int64)
    batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * wt_tokens)
    batched_sequences_to_run_with_masks += padded_masked_tensor * mask_token # add masks                
    

    from_mutation = dataset.one_hot_mut_info[0]
    to_mutation = dataset.one_hot_mut_info[1]
    

    full_msa = read_msa("%s/%s" % (MSA_PATH, STOCK_MSAS[3]))    
    labels, batch_strs, full_msa_tokens = msa_converter(full_msa)
    full_msa_tokens = torch.cat([full_msa_tokens, torch.ones((1, full_msa_tokens.shape[1], 1)) * eos_token], dim=2)
    
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(trainable_msa_transformer.parameters(), lr=3e-4, weight_decay=0.1)
    loss = torch.nn.CrossEntropyLoss()
            
    #train_path
    
    
    for epoch in range(0, 20):
        for data_iter_step, batch in enumerate(train_data_loader):
            
            if data_iter_step % 30 == 0:
                evaulate_msa_transformer_all_masks(trainable_msa_transformer, 
                                                   esm2_alphabet, 
                                                   dataset2, 
                                                   full_msa_tokens, 
                                                   wt_tokens,
                                                   dataset2.used_masks_tensor)
                
                
            # training loop
            trainable_msa_transformer.train()
    
            sampled = torch.tensor([0] + sample(range(1,1019), 0)) # ignore 
            # msa = [m for i,m in enumerate(full_msa) if i in sampled]
            # msa_converter = esm2_alphabet.get_batch_converter()        
            # labels, batch_strs, msa_tokens = msa_converter(msa)
            
            msa_tokens_with_mask = torch.cat([batched_sequences_to_run_with_masks.view((1,1,-1)), full_msa_tokens[:,sampled,]], dim=1).to(torch.int64)
            
            # with torch.no_grad():
            #     ref_logits = ref_msa_transformer(msa_tokens_with_mask)
            #     ref_masked_logits = ref_logits["logits"][:,0,:,:]
            #     ref_masked_logits_repeat = ref_masked_logits[padded_mutated_positions.view((1,-1)),:].repeat(BATCH_SIZE,1,1)
            #     ref_masked_logits_repeat = einops.rearrange(ref_masked_logits_repeat, 'B S C -> B C S')
                
                           
            optimizer.zero_grad()
            logits = trainable_msa_transformer(msa_tokens_with_mask)
            masked_logits = logits["logits"][:,0,:,:]
                    
            if data_iter_step % 10 == 0:
                pssm = masked_logits.softmax(dim=2)
                fixed_pssm = pssm[:,1:-1,:].view((len(jonathans_reference_sequence), -1))
                
                predicted_fitness =\
                    fitness_from_prob_non_dms(fixed_pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
                    from_mutation[:,mutated_positions.view((-1))][0],
                    to_mutation[:,mutated_positions.view((-1))])
                
                ina = predicted_fitness[dataset.labels == 1].detach().numpy()
                act = predicted_fitness[dataset.labels == 0].detach().numpy()
                
                plot_hists(act, ina)
            
            masked_logits_repeat = masked_logits[padded_mutated_positions.view((1,-1)),:].repeat(BATCH_SIZE,1,1)
            masked_logits_repeat = einops.rearrange(masked_logits_repeat, 'B S C -> B C S')
            positive_generations = batch[0][:,padded_mutated_positions]
            negative_generations = batch[1][:,padded_mutated_positions]
            
            nll_loss = loss(masked_logits_repeat, positive_generations)
            
            log_probs = F.log_softmax(masked_logits_repeat, dim=1)
            #ref_log_probs = F.log_softmax(ref_masked_logits_repeat, dim=1)
            
            preferred_log_probs = torch.gather(log_probs,
                                               1, 
                                               positive_generations.unsqueeze(dim=1))
            
            unpreferred_log_probs = torch.gather(log_probs,
                                                 1, 
                                                 negative_generations.unsqueeze(dim=1))
            
            # preferred_ref_log_probs = torch.gather(ref_log_probs,
            #                                        1, 
            #                                        positive_generations.unsqueeze(dim=1))
            
            # unpreferred_ref_log_probs = torch.gather(ref_log_probs,
            #                                          1, 
            #                                          negative_generations.unsqueeze(dim=1))
            
            
            
            # log_diff = dpo_beta * (preferred_log_probs - preferred_ref_log_probs) -\
            #            dpo_beta * (unpreferred_log_probs - unpreferred_ref_log_probs)
                       
                       
            #log_diff = dpo_beta * (preferred_log_probs - unpreferred_log_probs)# -\
                       #dpo_beta * (unpreferred_log_probs - unpreferred_ref_log_probs)                       
            
            #dpo_loss = -F.logsigmoid(log_diff).mean()
            dpo_loss = -F.logsigmoid(preferred_log_probs - unpreferred_log_probs).mean()
            
            total_loss = dpo_loss#nll_loss + dpo_loss
            total_loss.backward()
            print("Loss (%.3f [DPO:%.3f, NLL:%.3f]) [Epoch %d, I %d]" %\
                  (total_loss.item(),
                   dpo_loss.item(),
                   nll_loss.item(),
                   epoch, 
                   data_iter_step))
            optimizer.step()

    save_weights = True


def evaulate_msa_transformer_all_masks(model, 
                                       alphabet, 
                                       dataset, 
                                       full_msa_tokens, 
                                       wt_tokens,
                                       provided_masks=None):
    
    if provided_masks is None:
        all_masks = torch.unique(dataset.one_hot_mut_info[2], dim=0) 
    else:
        all_masks = provided_masks
    
    
    B, S = all_masks.size()
    
    # all_masks = all_masks[torch.tensor(sample(range(B), 80)),:]
    # B, S = all_masks.size()
    
    pad=torch.tensor(0).view((1,-1)).repeat(B,1)
    padded_masked_tensor = torch.cat([pad, all_masks, pad], dim=1)
    
    padded_mutated_positions = (padded_masked_tensor == 1)
    mask_token  = torch.tensor(alphabet.encode("<mask>"), dtype=torch.int64)
    
    batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * wt_tokens)
    batched_sequences_to_run_with_masks += padded_masked_tensor * mask_token # add masks                
    
    
    
    MINI_BATCH_SIZE = 1
    N_BATCHES = B
    
    all_pssms = torch.tensor([])
    print("Evaluating.")
    for mini_batch_idx in range(N_BATCHES):
        
        s_i = mini_batch_idx * MINI_BATCH_SIZE
        e_i = (mini_batch_idx + 1) * MINI_BATCH_SIZE
        
        batch_size = MINI_BATCH_SIZE
        
        # if mini_batch_idx == (N_BATCHES - 1):
        #     e_i = B
        #     batch_size = e_i - s_i
        
        if mini_batch_idx % 5 == 0: 
            print("\t%d - %d" % (s_i, e_i))
        
        sampled = torch.tensor([0] + sample(range(1,1019), 0))
        
        msa_tokens_with_mask = torch.cat([batched_sequences_to_run_with_masks[s_i:e_i].view((1,batch_size,-1)),
                                          full_msa_tokens[:,sampled,:]], dim=1).to(torch.int64)
        
        
        with torch.no_grad():
            logits = model(msa_tokens_with_mask)
            masked_logits = logits["logits"][:,0,:,:]            
            pssms = logits["logits"][:,0:batch_size,:,:].squeeze(dim=0).softmax(dim=2)
            all_pssms = torch.cat([all_pssms, pssms], dim=0)
        
    
    gt_labels = dataset.labels
    
    act = torch.tensor([])
    ina = torch.tensor([])
    
    for pssm_idx in range(N_BATCHES):
        mask = all_masks[pssm_idx,:]
        mutated_positions = mask == 1
        indices_of_mask  = torch.where((dataset.one_hot_mut_info[2] == mask).sum(dim=1) == 238)[0]
        
        pssm = all_pssms[pssm_idx]
        fixed_pssm = pssm[1:-1,:].view((S, -1))
        from_mutation = dataset.one_hot_mut_info[0][indices_of_mask,:]
        to_mutation = dataset.one_hot_mut_info[1][indices_of_mask,:]
        
        predicted_fitness =\
                    fitness_from_prob_non_dms(fixed_pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
                    from_mutation[:,mutated_positions.view((-1))][0],
                    to_mutation[:,mutated_positions.view((-1))])
                    
        act = torch.cat([act, predicted_fitness[gt_labels[indices_of_mask] == 0]], dim=0)
        ina = torch.cat([ina, predicted_fitness[gt_labels[indices_of_mask] == 1]], dim=0)
                
        
    plot_hists(act, ina)
        
def train_msa_transformer_advanced_mask():
    
    dpo_beta = 5
    BATCH_SIZE = 1
    root_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/"
    base_path = "%s/data/configuration/" % root_path
    train_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq_train.csv" % base_path
    saved_weights_path = "/%s/itayFold/weights/esm2/retrained/msa_transformer" % root_path
    
    model_name = "esm_msa1b_t12_100M_UR50S"
    trainable_msa_transformer, esm2_alphabet  = load_esm2_model_and_alphabet(model_name)
    ref_msa_transformer, esm2_alphabet  = load_esm2_model_and_alphabet(model_name)
    msa_converter = esm2_alphabet.get_batch_converter()     
    
    for p in ref_msa_transformer.parameters(): p.requires_grad = False
    
    jonathans_reference_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"



    dataset = \
        Esm2SequenceActivityContrastiveDatasetAdvancedMask(train_path,
                                               esm2_alphabet,
                                               get_mutated_position_function=get_mutated_position_function_gfp_n2,
                                               cache=True,
                                               model_name="msa_transformer",
                                               activity_column_name='is_unsorted',
                                               sequence_column_name="FullSeq",
                                               ref_seq=jonathans_reference_sequence,
                                               labels_dtype=torch.int64)
    
    
    #masked_tensor = torch.unique(dataset.one_hot_mut_info[2], dim=0)
    masked_tensor = dataset.wt_one_hot#torch.unique(dataset.wt_one_hot, dim=0)
    pad=torch.tensor(0).view((1,-1))
    padded_masked_tensor = torch.cat([pad, masked_tensor.view((1,-1)), pad], dim=1)
        
    wt_padded_mutated_positions = (padded_masked_tensor == 1).view(-1)
    mutated_positions = masked_tensor == 1
    
    wt_tokens = torch.tensor(esm2_alphabet.encode("<cls>" + jonathans_reference_sequence + "<eos>"), dtype=torch.int64).view((1,-1))
    eos_token = torch.tensor(esm2_alphabet.encode("<eos>"), dtype=torch.int64)
    mask_token = torch.tensor(esm2_alphabet.encode("<mask>"), dtype=torch.int64)
    
    
    from_mutation = dataset.one_hot_mut_info[0]
    to_mutation = dataset.one_hot_mut_info[1]
    

    full_msa = read_msa("%s/%s" % (MSA_PATH, STOCK_MSAS[3]))    
    labels, batch_strs, full_msa_tokens = msa_converter(full_msa)
    full_msa_tokens = torch.cat([full_msa_tokens, torch.ones((1, full_msa_tokens.shape[1], 1)) * eos_token], dim=2)
    
    train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = torch.optim.Adam(trainable_msa_transformer.parameters(), lr=3e-4, weight_decay=0.1)
    loss = torch.nn.CrossEntropyLoss()
             
    total_counter = 0
    for epoch in range(0, 5000):
        for data_iter_step, batch in enumerate(train_data_loader):
            # training loop
                        
            if total_counter % 500 == 0:
                evaulate_msa_transformer_all_masks(trainable_msa_transformer, 
                                                   esm2_alphabet, 
                                                   dataset, 
                                                   full_msa_tokens, 
                                                   wt_tokens,
                                                   dataset.used_masks_tensor)
            
            total_counter += 1
            
            trainable_msa_transformer.train()            
            positives = batch[0]
            negatives = batch[1]
            one_hot_positions = batch[2]
            
            _, B, S = positives.size()
            
            pad=torch.tensor(0).view((1,-1)).repeat(BATCH_SIZE, 1)
            padded_masked_tensor = torch.cat([pad, one_hot_positions, pad], dim=1)
            padded_mutated_positions = (padded_masked_tensor == 1)
            
            batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * positives[0,0,:])
            batched_sequences_to_run_with_masks += padded_masked_tensor * mask_token # add masks                
            
            sampled = torch.tensor([0] + sample(range(1,1019), 0)) # ignore 
            #msa = [m for i,m in enumerate(full_msa) if i in sampled]                 
            #torch.cat([msa_tokens, torch.ones((1, msa_tokens.shape[1], 1)) * eos_token], dim=2)
            
            msa_tokens_with_mask = torch.cat([batched_sequences_to_run_with_masks.view((1,BATCH_SIZE,-1)), 
                                              full_msa_tokens[:,sampled,:]], 
                                             dim=1).to(torch.int64)
            
            # with torch.no_grad():
            #     ref_logits = ref_msa_transformer(msa_tokens_with_mask)
            #     ref_masked_logits = ref_logits["logits"][:,0,:,:].squeeze(dim=0)
            #     ref_masked_logits_repeat = ref_masked_logits[padded_mutated_positions.view(-1),:].repeat(B,1,1)
            #     ref_masked_logits_repeat = einops.rearrange(ref_masked_logits_repeat, 'B S C -> B C S')
                       
                
            optimizer.zero_grad()
            logits = trainable_msa_transformer(msa_tokens_with_mask)
            masked_logits = logits["logits"][:,0,:,:].squeeze(dim=0)
            masked_logits_repeat = masked_logits[padded_mutated_positions.view((-1)),:].repeat(positives.shape[1],1,1)
            masked_logits_repeat = einops.rearrange(masked_logits_repeat, 'B S C -> B C S')
            
            
            
            positive_generations = positives.squeeze(dim=0)[:,padded_mutated_positions.squeeze(dim=0)]
            negative_generations = negatives.squeeze(dim=0)[:,padded_mutated_positions.squeeze(dim=0)]
            
            nll_loss = loss(einops.rearrange(masked_logits[wt_padded_mutated_positions.view((-1)),:].repeat(positives.shape[1],1,1), 'B S C -> B C S'), 
                            positives.squeeze(dim=0)[:,wt_padded_mutated_positions.squeeze(dim=0)])
            
            log_probs = F.log_softmax(masked_logits, dim=1)
            #ref_log_probs = F.log_softmax(ref_masked_logits, dim=1)
            
            preferred_log_probs = torch.gather(log_probs[padded_mutated_positions.squeeze(dim=0)].repeat((B,1)),
                                               1, 
                                               positive_generations)
            
            unpreferred_log_probs = torch.gather(log_probs[padded_mutated_positions.squeeze(dim=0)].repeat((B,1)),
                                                 1, 
                                                 negative_generations)
            
            # preferred_ref_log_probs = torch.gather(ref_log_probs[padded_mutated_positions.squeeze(dim=0)].repeat((B,1)),
            #                                        1, 
            #                                        positive_generations)
            
            # unpreferred_ref_log_probs = torch.gather(ref_log_probs[padded_mutated_positions.squeeze(dim=0)].repeat((B,1)),
            #                                          1, 
            #                                          negative_generations)
            
            
            
            # log_diff = dpo_beta * (preferred_log_probs - preferred_ref_log_probs) -\
            #            dpo_beta * (unpreferred_log_probs - unpreferred_ref_log_probs)
                       
                       
            #log_diff = dpo_beta * (preferred_log_probs - unpreferred_log_probs)# -\
                       #dpo_beta * (unpreferred_log_probs - unpreferred_ref_log_probs)                       
            
            #dpo_loss = -F.logsigmoid(log_diff).mean()
            #dpo_loss = -F.logsigmoid(preferred_log_probs - unpreferred_log_probs).mean()
            dpo_loss = -F.logsigmoid(dpo_beta * (preferred_log_probs - unpreferred_log_probs)).mean()
            
            total_loss = dpo_loss + nll_loss
            total_loss.backward()
            print("Loss (%.3f [DPO:%.3f, NLL:%.3f]) [Epoch %d, I %d]" %\
                  (total_loss.item(),
                   dpo_loss.item(),
                   nll_loss.item(),
                   epoch, 
                   data_iter_step))
            optimizer.step()
            
            
    model_params = trainable_msa_transformer.state_dict()
    optimizer_params = optimizer.state_dict()
    saved_dict = {"model_params":model_params,
                  "optimizer_params":optimizer_params}
    
    torch.save(saved_dict, "%s/partially_masked_full_nll_20_24.pt" % saved_weights_path)
    save_weights = True

    


def test_msa_transformer_regular_mask():
    
    dpo_beta = .5
    BATCH_SIZE = 256
    root_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/"
    base_path = "%s/data/configuration/" % root_path
    test_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq_test.csv" % base_path
    saved_weights_path = "/%s/itayFold/weights/esm2/retrained/msa_transformer" % root_path
    
    model_name = "esm_msa1b_t12_100M_UR50S"
    msa_transformer, esm2_alphabet  = load_esm2_model_and_alphabet(model_name)
    msa_converter = esm2_alphabet.get_batch_converter()     
    
    params = torch.load("%s/partially_masked_full_nll.pt" % saved_weights_path)
    msa_transformer.load_state_dict(params["model_params"])
    
    
    
    jonathans_reference_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"


    # dataset2 = \
    #     Esm2SequenceActivityContrastiveDatasetAdvancedMask(train_path,
    #                                            esm2_alphabet,
    #                                            get_mutated_position_function=get_mutated_position_function_gfp_n2,
    #                                            cache=True,
    #                                            model_name="msa_transformer",
    #                                            activity_column_name='is_unsorted',
    #                                            sequence_column_name="FullSeq",
    #                                            ref_seq=jonathans_reference_sequence,
    #                                            labels_dtype=torch.int64)

    dataset = \
        Esm2SequenceActivityDataset(test_path,
                                    esm2_alphabet,
                                    get_mutated_position_function=get_mutated_position_function_gfp,
                                    cache=True,
                                    model_name="msa_transformer",
                                    activity_column_name='is_unsorted',
                                    sequence_column_name="FullSeq",
                                    ref_seq=jonathans_reference_sequence,
                                    labels_dtype=torch.int64)
    
    
    #masked_tensor = torch.unique(dataset.one_hot_mut_info[2], dim=0)
    masked_tensor = dataset.wt_one_hot#torch.unique(dataset.wt_one_hot, dim=0)
    pad=torch.tensor(0).view((1,-1))
    padded_masked_tensor = torch.cat([pad, masked_tensor.view((1,-1)), pad], dim=1)
        
    padded_mutated_positions = (padded_masked_tensor == 1).view(-1)
    mutated_positions = masked_tensor == 1
    
    wt_tokens = torch.tensor(esm2_alphabet.encode("<cls>" + jonathans_reference_sequence + "<eos>"), dtype=torch.int64).view((1,-1))
    eos_token = torch.tensor(esm2_alphabet.encode("<eos>"), dtype=torch.int64)    
    mask_token = torch.tensor(esm2_alphabet.encode("<mask>"), dtype=torch.int64)
    batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * wt_tokens)
    batched_sequences_to_run_with_masks += padded_masked_tensor * mask_token # add masks                
    

    from_mutation = dataset.one_hot_mut_info[0]
    to_mutation = dataset.one_hot_mut_info[1]
    

    full_msa = read_msa("%s/%s" % (MSA_PATH, STOCK_MSAS[3]))    
    labels, batch_strs, full_msa_tokens = msa_converter(full_msa)
    full_msa_tokens = torch.cat([full_msa_tokens, torch.ones((1, full_msa_tokens.shape[1], 1)) * eos_token], dim=2)
    
            
    #train_path
    
    
    for epoch in range(0, 20):
        for data_iter_step, batch in enumerate(train_data_loader):
            
            if data_iter_step % 30 == 0:
                evaulate_msa_transformer_all_masks(trainable_msa_transformer, 
                                                   esm2_alphabet, 
                                                   dataset2, 
                                                   full_msa_tokens, 
                                                   wt_tokens,
                                                   dataset2.used_masks_tensor)
                
                
            # training loop
            trainable_msa_transformer.train()
    
            sampled = torch.tensor([0] + sample(range(1,1019), 0)) # ignore 
            # msa = [m for i,m in enumerate(full_msa) if i in sampled]
            # msa_converter = esm2_alphabet.get_batch_converter()        
            # labels, batch_strs, msa_tokens = msa_converter(msa)
            
            msa_tokens_with_mask = torch.cat([batched_sequences_to_run_with_masks.view((1,1,-1)), full_msa_tokens[:,sampled,]], dim=1).to(torch.int64)
            
            # with torch.no_grad():
            #     ref_logits = ref_msa_transformer(msa_tokens_with_mask)
            #     ref_masked_logits = ref_logits["logits"][:,0,:,:]
            #     ref_masked_logits_repeat = ref_masked_logits[padded_mutated_positions.view((1,-1)),:].repeat(BATCH_SIZE,1,1)
            #     ref_masked_logits_repeat = einops.rearrange(ref_masked_logits_repeat, 'B S C -> B C S')
                
                           
            optimizer.zero_grad()
            logits = trainable_msa_transformer(msa_tokens_with_mask)
            masked_logits = logits["logits"][:,0,:,:]
                    
            if data_iter_step % 10 == 0:
                pssm = masked_logits.softmax(dim=2)
                fixed_pssm = pssm[:,1:-1,:].view((len(jonathans_reference_sequence), -1))
                
                predicted_fitness =\
                    fitness_from_prob_non_dms(fixed_pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
                    from_mutation[:,mutated_positions.view((-1))][0],
                    to_mutation[:,mutated_positions.view((-1))])
                
                ina = predicted_fitness[dataset.labels == 1].detach().numpy()
                act = predicted_fitness[dataset.labels == 0].detach().numpy()
                
                plot_hists(act, ina)
            
            masked_logits_repeat = masked_logits[padded_mutated_positions.view((1,-1)),:].repeat(BATCH_SIZE,1,1)
            masked_logits_repeat = einops.rearrange(masked_logits_repeat, 'B S C -> B C S')
            positive_generations = batch[0][:,padded_mutated_positions]
            negative_generations = batch[1][:,padded_mutated_positions]
            
            nll_loss = loss(masked_logits_repeat, positive_generations)
            
            log_probs = F.log_softmax(masked_logits_repeat, dim=1)
            #ref_log_probs = F.log_softmax(ref_masked_logits_repeat, dim=1)
            
            preferred_log_probs = torch.gather(log_probs,
                                               1, 
                                               positive_generations.unsqueeze(dim=1))
            
            unpreferred_log_probs = torch.gather(log_probs,
                                                 1, 
                                                 negative_generations.unsqueeze(dim=1))
            
            # preferred_ref_log_probs = torch.gather(ref_log_probs,
            #                                        1, 
            #                                        positive_generations.unsqueeze(dim=1))
            
            # unpreferred_ref_log_probs = torch.gather(ref_log_probs,
            #                                          1, 
            #                                          negative_generations.unsqueeze(dim=1))
            
            
            
            # log_diff = dpo_beta * (preferred_log_probs - preferred_ref_log_probs) -\
            #            dpo_beta * (unpreferred_log_probs - unpreferred_ref_log_probs)
                       
                       
            #log_diff = dpo_beta * (preferred_log_probs - unpreferred_log_probs)# -\
                       #dpo_beta * (unpreferred_log_probs - unpreferred_ref_log_probs)                       
            
            #dpo_loss = -F.logsigmoid(log_diff).mean()
            dpo_loss = -F.logsigmoid(preferred_log_probs - unpreferred_log_probs).mean()
            
            total_loss = dpo_loss#nll_loss + dpo_loss
            total_loss.backward()
            print("Loss (%.3f [DPO:%.3f, NLL:%.3f]) [Epoch %d, I %d]" %\
                  (total_loss.item(),
                   dpo_loss.item(),
                   nll_loss.item(),
                   epoch, 
                   data_iter_step))
            optimizer.step()

    save_weights = True


def test_msa_transformer_full_partial_mask():    
    dpo_beta = .5
    BATCH_SIZE = 256
    root_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/"
    base_path = "%s/data/configuration/" % root_path
    test_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq_test.csv" % base_path
    train_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq_train.csv" % base_path
    saved_weights_path = "/%s/itayFold/weights/esm2/retrained/msa_transformer" % root_path
    
    model_name = "esm2_t33_650M_UR50D"
    msa_transformer, esm2_alphabet  = load_esm2_model_and_alphabet(model_name)
    msa_converter = esm2_alphabet.get_batch_converter()     
    
    #params = torch.load("%s/dpo_loss_only_train_all_masked.pt" % saved_weights_path)
    #msa_transformer.load_state_dict(params["model_params"])
    
    
    
    jonathans_reference_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"


    wt_tokens = torch.tensor(esm2_alphabet.encode("<cls>" + jonathans_reference_sequence + "<eos>"), dtype=torch.int64).view((1,-1))
    eos_token = torch.tensor(esm2_alphabet.encode("<eos>"), dtype=torch.int64)    
    mask_token = torch.tensor(esm2_alphabet.encode("<mask>"), dtype=torch.int64)
    
    dataset = \
        Esm2SequenceActivityTrainTest(train_path, 
                                      test_path, 
                                      msa_transformer,
                                      esm2_alphabet,
                                      full_mask_mut_positions=get_mutated_position_function_gfp,
                                      partial_mask_mut_positions=get_mutated_position_function_gfp_n2,
                                      use_full_mask_only=False,
                                      cache=True,
                                      model_name="esm2",
                                      activity_column_name='is_unsorted',
                                      sequence_column_name="FullSeq",
                                      ref_seq=jonathans_reference_sequence,
                                      labels_dtype=torch.int64)
        
        
    dataset.evaluate_full(is_msa_transformer=False)
    
    # batch_size is 1 as batches are internally implemented
    train_data_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=1, 
                                                    shuffle=True)
    optimizer = torch.optim.Adam(msa_transformer.parameters(), 
                                 lr=1e-5, 
                                 weight_decay=0.1)
    loss = torch.nn.CrossEntropyLoss()
    
    for epoch in range(0, 20):
        for data_iter_step, batch in enumerate(train_data_loader):
            # training loop
            msa_transformer.train()        
                
            positives = batch[0]
            negatives = batch[1]
            one_hot_positions = batch[2].view((1,1,-1))
            
            _, B, S = positives.size()
            
            pad=torch.tensor(0).view((1,1,1))
            padded_masked_tensor = torch.cat([pad, one_hot_positions, pad], dim=2)
            padded_mutated_positions = (padded_masked_tensor == 1)
            
            # which positive we choose doesnt matter as all the positives are masked in the same position -
            # run this for sanity:
            # for i in range(20):
            #    for j in range(20):
            #        if i != j:
            #            print((((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * positives[0,i,:]) ==  
            #                   ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * positives[0,j,:])).sum())
            
            batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * positives[0,0,:])
            batched_sequences_to_run_with_masks += padded_masked_tensor * mask_token # add masks                
            
            
            # IN CASE USING MSA FOR INFERENCE
            #sampled = torch.tensor([0] + sample(range(1,1019), 0)) # ignore 
            # msa = [m for i,m in enumerate(full_msa) if i in sampled]
            # msa_converter = esm2_alphabet.get_batch_converter()        
            # labels, batch_strs, msa_tokens = msa_converter(msa)
            
            #msa_tokens_with_mask = torch.cat([batched_sequences_to_run_with_masks.view((1,1,-1)), full_msa_tokens[:,sampled,]], dim=1).to(torch.int64)
               
            optimizer.zero_grad()
            logits = msa_transformer(batched_sequences_to_run_with_masks)
            masked_logits = logits["logits"][:,0,:,:]
            
            # DPO LOSS WITH REF 
            # with torch.no_grad():
            #     ref_logits = ref_msa_transformer(msa_tokens_with_mask)
            #     ref_masked_logits = ref_logits["logits"][:,0,:,:]
            #     ref_masked_logits_repeat = ref_masked_logits[padded_mutated_positions.view((1,-1)),:].repeat(BATCH_SIZE,1,1)
            #     ref_masked_logits_repeat = einops.rearrange(ref_masked_logits_repeat, 'B S C -> B C S')
                    
            if data_iter_step % 10 == 0:
                dataset.evaluate_full(is_msa_transformer=False)
            
            masked_logits_repeat = masked_logits[padded_mutated_positions.view((1,-1)),:].repeat(B,1,1)
            masked_logits_repeat = einops.rearrange(masked_logits_repeat, 'B S C -> B C S')
            positive_generations = positives[:,:,padded_mutated_positions.view(-1)].squeeze(dim=0)
            negative_generations = negatives[:,:,padded_mutated_positions.view(-1)].squeeze(dim=0)
            
            print(masked_logits_repeat.softmax(dim=1).argmax(dim=1)[0])
            nll_loss = loss(masked_logits_repeat, positive_generations)
            
            probs =  masked_logits_repeat.softmax(dim=1)
            log_probs = probs.log()
            #log_probs = F.log_softmax(masked_logits_repeat, dim=1)
            #ref_log_probs = F.log_softmax(ref_masked_logits_repeat, dim=1)
            
            preferred_log_probs = torch.gather(log_probs,
                                               1, 
                                               positive_generations.unsqueeze(dim=1))
            
            unpreferred_log_probs = torch.gather(log_probs,
                                                 1, 
                                                 negative_generations.unsqueeze(dim=1))
            
            
            preferred_probs = torch.gather(probs,
                                           1, 
                                           positive_generations.unsqueeze(dim=1))
            
            unpreferred_probs = torch.gather(probs,
                                             1, 
                                             negative_generations.unsqueeze(dim=1))
            
            # preferred_ref_log_probs = torch.gather(ref_log_probs,
            #                                        1, 
            #                                        positive_generations.unsqueeze(dim=1))
            
            # unpreferred_ref_log_probs = torch.gather(ref_log_probs,
            #                                          1, 
            #                                          negative_generations.unsqueeze(dim=1))
            
            
            
            # log_diff = dpo_beta * (preferred_log_probs - preferred_ref_log_probs) -\
            #            dpo_beta * (unpreferred_log_probs - unpreferred_ref_log_probs)
                       
                       
            #log_diff = dpo_beta * (preferred_log_probs - unpreferred_log_probs)# -\
                       #dpo_beta * (unpreferred_log_probs - unpreferred_ref_log_probs)                       
            
            #dpo_loss = -F.logsigmoid(log_diff).mean()
            dpo_loss = -F.logsigmoid(preferred_log_probs - unpreferred_log_probs).mean()
            
            odds_preferred = (preferred_probs.mean(dim=2)) / (1 - preferred_probs.mean(dim=2))
            odds_unpreferred = (unpreferred_probs.mean(dim=2)) / (1 - unpreferred_probs.mean(dim=2))
            orpo_loss = -F.logsigmoid((odds_preferred/odds_unpreferred).log()).mean()
            #total_loss = nll_loss + dpo_loss
            total_loss = nll_loss + orpo_loss
            total_loss.backward()
            print("Loss (%.3f [DPO:%.3f, NLL:%.3f, ORPO:%.3f]) [Epoch %d, I %d]" %\
                  (total_loss.item(),
                   dpo_loss.item(),
                   nll_loss.item(),
                   orpo_loss.item(),
                   epoch, 
                   data_iter_step))
            optimizer.step()

    save_weights = True


def test_esm2_full_partial_mask():       
    
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
    
    pre_load_weights = False
    mutations_to_include = [1,2]
    dpo_beta = .5
    BATCH_SIZE = 256
    project_name = "esm2_600m_nll_orpo_%s" % "_".join([str(m) for m in mutations_to_include])
    root_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/"
    base_path = "%s/data/configuration/" % root_path
    #test_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq_test.csv" % base_path
    dataset_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq.csv" % base_path
    saved_weights_path = "/%s/itayFold/weights/esm2/retrained/esm2_600m_experiments/" % root_path
    evaluation_path  = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/results/gfp_dataset/finetuned_models_evaluated"
    
    model_name = "esm2_t33_650M_UR50D"
    model, esm2_alphabet  = load_esm2_model_and_alphabet(model_name)
    model = model.to(device)
    #msa_converter = esm2_alphabet.get_batch_converter()     
    
    if pre_load_weights:
        params = torch.load("%s/%s.pt" % (saved_weights_path, project_name))
        model.load_state_dict(params["model_params"])
    
            
    jonathans_reference_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"


    wt_tokens = torch.tensor(esm2_alphabet.encode("<cls>" + jonathans_reference_sequence + "<eos>"), dtype=torch.int64, device=device).view((1,-1))
    eos_token = torch.tensor(esm2_alphabet.encode("<eos>"), dtype=torch.int64, device=device) 
    mask_token = torch.tensor(esm2_alphabet.encode("<mask>"), dtype=torch.int64, device=device)        
    
    dataset = \
        Esm2SequenceActivityTrainTest(project_name,
                                      evaluation_path,
                                      dataset_path,
                                      lambda sdf: get_indices(sdf, mutations_to_include),
                                      lambda sdf: get_indices(sdf, mutations_to_include, rev=True),
                                      model,
                                      esm2_alphabet,
                                      full_mask_mut_positions=get_mutated_position_function_gfp,
                                      partial_mask_mut_positions=get_mutated_position_function_gfp_n2,
                                      use_partial_mask_only=False,
                                      cache=True,
                                      model_name="esm2",
                                      activity_column_name='is_unsorted',
                                      sequence_column_name="FullSeq",
                                      ref_seq=jonathans_reference_sequence,
                                      labels_dtype=torch.int64)
        
    
#    dataset.evaluate_across_masks()   
    #dataset.evaluate_full(is_msa_transformer=False)
    
    # batch_size is 1 as batches are internally implemented
    train_data_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=1, 
                                                    shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=1e-5, 
                                 weight_decay=0.1)
    loss = torch.nn.CrossEntropyLoss().to(device)
    
    separation = torch.tensor(0)
    
    for epoch in range(0, 2000):
        for data_iter_step, batch in enumerate(train_data_loader):
            # training loop
            model.train()        
                
            positives = batch[0].to(device)
            negatives = batch[1].to(device)
            one_hot_positions = batch[2].view((1,1,-1)).to(device)
            
            _, B, S = positives.size()
            
            pad=torch.tensor(0, device=device).view((1,1,1))
            padded_masked_tensor = torch.cat([pad, one_hot_positions, pad], dim=2)
            padded_mutated_positions = (padded_masked_tensor == 1)
            
            # which positive we choose doesnt matter as all the positives are masked in the same position -
            # run this for sanity:
            # for i in range(20):
            #    for j in range(20):
            #        if i != j:
            #            print((((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * positives[0,i,:]) ==  
            #                   ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64) - padded_masked_tensor) * positives[0,j,:])).sum())
            
            batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64, device=device) - padded_masked_tensor) * positives[0,0,:])
            batched_sequences_to_run_with_masks += padded_masked_tensor * mask_token # add masks                
            
            
            # IN CASE USING MSA FOR INFERENCE
            #sampled = torch.tensor([0] + sample(range(1,1019), 0)) # ignore 
            # msa = [m for i,m in enumerate(full_msa) if i in sampled]
            # msa_converter = esm2_alphabet.get_batch_converter()        
            # labels, batch_strs, msa_tokens = msa_converter(msa)
            
            #msa_tokens_with_mask = torch.cat([batched_sequences_to_run_with_masks.view((1,1,-1)), full_msa_tokens[:,sampled,]], dim=1).to(torch.int64)
               
            optimizer.zero_grad()
            logits = model(batched_sequences_to_run_with_masks.view((1,-1)))
            masked_logits = logits["logits"][0,:,:]
            
            # DPO LOSS WITH REF 
            # with torch.no_grad():
            #     ref_logits = ref_msa_transformer(msa_tokens_with_mask)
            #     ref_masked_logits = ref_logits["logits"][:,0,:,:]
            #     ref_masked_logits_repeat = ref_masked_logits[padded_mutated_positions.view((1,-1)),:].repeat(BATCH_SIZE,1,1)
            #     ref_masked_logits_repeat = einops.rearrange(ref_masked_logits_repeat, 'B S C -> B C S')
                    
            if data_iter_step % 20 == 0:
                evaluation_metric =\
                    dataset.evaluate_full(is_msa_transformer=False, 
                                          return_act_inact=True,
                                          device=device)
                new_separation = np.median(evaluation_metric[0]) - np.median(evaluation_metric[1])
                
                if new_separation > separation:
                    print("Saving new model (improved from %.3f to %.3f" % (separation, new_separation))
                    separation = new_separation
                    
                    model_params = model.state_dict()
                    optimizer_params = optimizer.state_dict()
                    saved_dict = {"model_params ":model_params,
                                  "optimizer_params":optimizer_params}
                    
                    torch.save(saved_dict, "%s/best_sep_%s.pt" % (saved_weights_path, project_name))
            
            masked_logits_repeat = masked_logits[padded_mutated_positions.view((-1)),:].repeat(B,1,1)
            masked_logits_repeat = einops.rearrange(masked_logits_repeat, 'B S C -> B C S')
            positive_generations = positives[:,:,padded_mutated_positions.view(-1)].squeeze(dim=0)
            negative_generations = negatives[:,:,padded_mutated_positions.view(-1)].squeeze(dim=0)
            
            print(masked_logits_repeat.softmax(dim=1).argmax(dim=1)[0])
            nll_loss = loss(masked_logits_repeat, positive_generations)
            
            probs =  masked_logits_repeat.softmax(dim=1)
            log_probs = probs.log()
            #log_probs = F.log_softmax(masked_logits_repeat, dim=1)
            #ref_log_probs = F.log_softmax(ref_masked_logits_repeat, dim=1)
            
            preferred_log_probs = torch.gather(log_probs,
                                               1, 
                                               positive_generations.unsqueeze(dim=1))
            
            unpreferred_log_probs = torch.gather(log_probs,
                                                 1, 
                                                 negative_generations.unsqueeze(dim=1))
            
            
            preferred_probs = torch.gather(probs,
                                           1, 
                                           positive_generations.unsqueeze(dim=1))
            
            unpreferred_probs = torch.gather(probs,
                                             1, 
                                             negative_generations.unsqueeze(dim=1))
            
            # preferred_ref_log_probs = torch.gather(ref_log_probs,
            #                                        1, 
            #                                        positive_generations.unsqueeze(dim=1))
            
            # unpreferred_ref_log_probs = torch.gather(ref_log_probs,
            #                                          1, 
            #                                          negative_generations.unsqueeze(dim=1))
            
            
            
            # log_diff = dpo_beta * (preferred_log_probs - preferred_ref_log_probs) -\
            #            dpo_beta * (unpreferred_log_probs - unpreferred_ref_log_probs)
                       
                       
            #log_diff = dpo_beta * (preferred_log_probs - unpreferred_log_probs)# -\
                       #dpo_beta * (unpreferred_log_probs - unpreferred_ref_log_probs)                       
            
            #dpo_loss = -F.logsigmoid(log_diff).mean()
            dpo_loss = -F.logsigmoid(preferred_log_probs - unpreferred_log_probs).mean()
            
            odds_preferred = (preferred_probs.mean(dim=2)) / (1 - preferred_probs.mean(dim=2))
            odds_unpreferred = (unpreferred_probs.mean(dim=2)) / (1 - unpreferred_probs.mean(dim=2))
            orpo_loss = -F.logsigmoid((odds_preferred/odds_unpreferred).log()).mean()
            total_loss = nll_loss + orpo_loss
            #total_loss = dpo_loss
            total_loss.backward()
            print("Loss (%.3f [DPO:%.3f, NLL:%.3f, ORPO:%.3f]) [Epoch %d, I %d]" %\
                  (total_loss.item(),
                   dpo_loss.item(),
                   nll_loss.item(),
                   orpo_loss.item(),
                   epoch, 
                   data_iter_step))
            optimizer.step()

    save_weights = True
    
    model_params = model.state_dict()
    optimizer_params = optimizer.state_dict()
    saved_dict = {"model_params ":model_params,
                  "optimizer_params":optimizer_params}
    
    torch.save(saved_dict, "%s/final_10k_%s.pt" % (saved_weights_path, project_name))


def test_esm2_full_partial_mask_dkl_loss():   
    
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
    
    pre_load_weights = False
    mutations_to_include = [1,2,3,4]
    dpo_beta = .5
    BATCH_SIZE = 256
    project_name = "esm2_600m_dkl_%s" % "_".join([str(m) for m in mutations_to_include])
    root_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/"
    base_path = "%s/data/configuration/" % root_path
    #test_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq_test.csv" % base_path
    dataset_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq.csv" % base_path
    saved_weights_path = "/%s/itayFold/weights/esm2/retrained/esm2_600m_experiments/" % root_path
    evaluation_path  = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/results/gfp_dataset/finetuned_models_evaluated"
    
    model_name = "esm2_t33_650M_UR50D"
    model, esm2_alphabet  = load_esm2_model_and_alphabet(model_name)
    model = model.to(device)
    #msa_converter = esm2_alphabet.get_batch_converter()     
    
    if pre_load_weights:
        params = torch.load("%s/%s.pt" % (saved_weights_path, project_name))
        model.load_state_dict(params["model_params"])
    
            
    jonathans_reference_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"
    len_ref_seq = len(jonathans_reference_sequence)

    wt_tokens = torch.tensor(esm2_alphabet.encode("<cls>" + jonathans_reference_sequence + "<eos>"), dtype=torch.int64, device=device).view((1,-1))
    eos_token = torch.tensor(esm2_alphabet.encode("<eos>"), dtype=torch.int64, device=device) 
    mask_token = torch.tensor(esm2_alphabet.encode("<mask>"), dtype=torch.int64, device=device)        
    
    dataset = \
        Esm2SequenceActivityTrainTest(project_name,
                                      evaluation_path,
                                      dataset_path,
                                      lambda sdf: get_indices(sdf, mutations_to_include),
                                      lambda sdf: get_indices(sdf, mutations_to_include, rev=True),
                                      model,
                                      esm2_alphabet,
                                      full_mask_mut_positions=get_mutated_position_function_gfp,
                                      partial_mask_mut_positions=get_mutated_position_function_gfp_n2,
                                      use_full_mask_only=True,
                                      mini_batch_size=50,
                                      cache=True,
                                      model_name="esm2",
                                      activity_column_name='is_unsorted',
                                      sequence_column_name="FullSeq",
                                      ref_seq=jonathans_reference_sequence,
                                      labels_dtype=torch.int64)
        
    
#    dataset.evaluate_across_masks()   
    #dataset.evaluate_full(is_msa_transformer=False)
    
    # batch_size is 1 as batches are internally implemented
    train_data_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=1, 
                                                    shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=1e-5, 
                                 weight_decay=0.1)
    
    loss = torch.nn.CrossEntropyLoss().to(device)
    dkl_loss = torch.nn.KLDivLoss().to(device)
    dataset.train_dataset_partial_mask.one_hot_mut_info =\
        dataset.train_dataset_partial_mask.one_hot_mut_info.to(device)
        
    dataset.train_dataset_full_mask.one_hot_mut_info =\
        dataset.train_dataset_full_mask.one_hot_mut_info.to(device)      
        
    separation = torch.tensor(0)
    
    for epoch in range(0, 2000):
        for data_iter_step, batch in enumerate(train_data_loader):
            # training loop
            model.train()        
                
            # positives = batch[0][0].to(device)
            # negatives = batch[0][1].to(device)
            # one_hot_positions = batch[0][2].view((1,1,-1)).to(device)
            # pair_indices = batch[0][3]
            # is_full = batch[1][0] == "full"
            
            
            positives = batch[0].to(device)
            negatives = batch[1].to(device)
            one_hot_positions = batch[2].view((1,1,-1)).to(device)
            pair_indices = batch[3]
            is_full = True
            
            _, B, S = positives.size()
            
            pad=torch.tensor(0, device=device).view((1,1,1))
            padded_masked_tensor = torch.cat([pad, one_hot_positions, pad], dim=2)
            padded_mutated_positions = (padded_masked_tensor == 1)
            
    
            batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64, device=device) - padded_masked_tensor) * positives[0,0,:])
            batched_sequences_to_run_with_masks += padded_masked_tensor * mask_token # add masks                
            
            
            optimizer.zero_grad()
            logits = model(batched_sequences_to_run_with_masks.view((1,-1)))
            masked_logits = logits["logits"][0,:,:]
                    
            pssm = masked_logits[1:-1,:].softmax(dim=1).view((len_ref_seq, -1))
                    
            
            pos_indices = torch.unique(pair_indices[:,:,0].view(-1))
            neg_indices = torch.unique(pair_indices[:,:,1].view(-1))
            indices = torch.cat([pos_indices,neg_indices]).to(device)
            pos_labels = torch.tensor(1, device=device).repeat(pos_indices.shape[0])
            neg_labels = torch.tensor(0, device=device).repeat(neg_indices.shape[0])
            labels = torch.cat([pos_labels, neg_labels])
            
            if is_full:
                relevant_one_hot  = dataset.train_dataset_full_mask.one_hot_mut_info
            else:
                relevant_one_hot  = dataset.train_dataset_partial_mask.one_hot_mut_info
                
            mutated_positions = one_hot_positions == 1
            
            from_mutation = relevant_one_hot[0][indices,:]
            to_mutation = relevant_one_hot[1][indices,:]
        
            predicted_fitness =\
                fitness_from_prob_non_dms(pssm[mutated_positions.view(-1)],#pssm[mutated_positions],
                                          from_mutation[:,mutated_positions.view((-1))][0],
                                          to_mutation[:,mutated_positions.view((-1))],
                                          device=device)
                        

            masked_logits_repeat = masked_logits[padded_mutated_positions.view((-1)),:].repeat(B,1,1)
            masked_logits_repeat = einops.rearrange(masked_logits_repeat, 'B S C -> B C S')
            positive_generations = positives[:,:,padded_mutated_positions.view(-1)].squeeze(dim=0)
            negative_generations = negatives[:,:,padded_mutated_positions.view(-1)].squeeze(dim=0)
            
            print(masked_logits_repeat.softmax(dim=1).argmax(dim=1)[0])
            nll_loss = loss(masked_logits_repeat, positive_generations)
            dkl = -dkl_loss(predicted_fitness,labels.to(torch.float32))
            
            total_loss = dkl + nll_loss
            #total_loss = dpo_loss
            total_loss.backward()
            print("Loss (%.3f [DKL:%.3f, NLL:%.3f]) [Epoch %d, I %d]" %\
                  (total_loss.item(),
                   dkl.item(),
                   nll_loss.item(),
                   epoch, 
                   data_iter_step))
                
            optimizer.step()
            
            if data_iter_step % 20 == 0:
                evaluation_metric =\
                    dataset.evaluate_full(is_msa_transformer=False, 
                                          return_act_inact=True,
                                          device=device)
                new_separation = np.median(evaluation_metric[0]) - np.median(evaluation_metric[1])
                
                if new_separation > separation:
                    print("Saving new model (improved from %.3f to %.3f" % (separation, new_separation))
                    separation = new_separation
                    
                    model_params = model.state_dict()
                    optimizer_params = optimizer.state_dict()
                    saved_dict = {"model_params ":model_params,
                                  "optimizer_params":optimizer_params}
                    
                    torch.save(saved_dict, "%s/best_sep_%s.pt" % (saved_weights_path, project_name))
        


    save_weights = True
    
    model_params = model.state_dict()
    optimizer_params = optimizer.state_dict()
    saved_dict = {"model_params ":model_params,
                  "optimizer_params":optimizer_params}
    
    torch.save(saved_dict, "%s/final_10k_%s.pt" % (saved_weights_path, project_name))




def evaluate():  
    device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
    
    pre_load_weights = True
    mutations_to_include = [1,2,3,4]
    dpo_beta = .5
    BATCH_SIZE = 256
    project_name = "esm2_600m_orpo_nll_%s" % "_".join([str(m) for m in mutations_to_include])
    root_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/"
    base_path = "%s/data/configuration/" % root_path
    #test_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq_test.csv" % base_path
    dataset_path = "%s/fixed_unique_gfp_sequence_dataset_full_seq.csv" % base_path
    saved_weights_path = "/%s/itayFold/weights/esm2/retrained/esm2_600m_experiments/" % root_path
    evaluation_path  = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/results/gfp_dataset/finetuned_models_evaluated"
    
    model_name = "esm2_t33_650M_UR50D"
    model, esm2_alphabet  = load_esm2_model_and_alphabet(model_name)
    model = model.to(device)
    #msa_converter = esm2_alphabet.get_batch_converter()     
    
    if pre_load_weights:
        params = torch.load("%s/best_sep_%s.pt" % (saved_weights_path, project_name))
        model.load_state_dict(params["model_params "])
    
            
    jonathans_reference_sequence = "MSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKRHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYN"


    wt_tokens = torch.tensor(esm2_alphabet.encode("<cls>" + jonathans_reference_sequence + "<eos>"), dtype=torch.int64, device=device).view((1,-1))
    eos_token = torch.tensor(esm2_alphabet.encode("<eos>"), dtype=torch.int64, device=device) 
    mask_token = torch.tensor(esm2_alphabet.encode("<mask>"), dtype=torch.int64, device=device)        
    
    dataset = \
        Esm2SequenceActivityTrainTest(project_name,
                                      evaluation_path,
                                      dataset_path,
                                      lambda sdf: get_indices(sdf, mutations_to_include),
                                      lambda sdf: get_indices(sdf, mutations_to_include, rev=True),
                                      model,
                                      esm2_alphabet,
                                      full_mask_mut_positions=get_mutated_position_function_gfp,
                                      partial_mask_mut_positions=get_mutated_position_function_gfp_n2,
                                      use_full_mask_only=False,
                                      cache=True,
                                      model_name="esm2",
                                      activity_column_name='is_unsorted',
                                      sequence_column_name="FullSeq",
                                      ref_seq=jonathans_reference_sequence,
                                      labels_dtype=torch.int64)
        
    
    dataset.evaluate_across_masks(device=device)   
    #dataset.evaluate_full(is_msa_transformer=False)

test_esm2_full_partial_mask_dkl_loss()
#test_esm2_full_partial_mask()
#evaluate()