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
import yaml
import argparse


from esm_smart_dataset import *
from sequence_space_utils import *

from Bio import pairwise2
from Bio.Seq import Seq
from Bio.Align import substitution_matrices
from Bio import SeqIO

from random import sample
from math import ceil

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
        raise BaseException("Unsupported model %s, model must be in: %s" %\
                              (model_name, ", ".join(supported_esm2_models)))
        
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
        

def get_indices(sequence_df, nmuts, nmuts_column="num_of_muts", rev=False, verbose=False):
        
    indices = np.repeat(False, sequence_df.shape[0])
        
    if type(nmuts) == int:
        nmuts = [nmuts]
            
    for nm in nmuts:
        indices = indices | (sequence_df[nmuts_column] == nm).to_numpy()
        if verbose:
            print("Indices included: %d" % sum(indices))
            
    if rev:
        indices = ~indices
            
    return(np.where(indices)[0].tolist())




def validate_args(\
                    dataset_path=None,
                    save_path=None,
                    device="cpu",
                    model_name="esm2_t33_650M_UR50D",
                    pretrained_weights_path=None,
                    loss=["nll", "orpo", "dkl"],
                    loss_weights={'dkl': 1, 'orpo': 1, 'nll': 1},
                    learning_rate=1e-5,
                    weight_decay=0.1,
                    indices=None,
                    batch_size=20,
                    iterations=20000,
                    checkpoint_every=5000,
                    best_sep_checkpoint=20,
                    mask_type="Both",   
                    ref_seq=None,
                    full_mask_pos_func=get_mutated_position_function_gfp,
                    partial_mask_pos_func=get_mutated_position_function_gfp_n2,
                    num_muts_column_name="num_muts",
                    activity_column_name='inactive',
                    sequence_column_name="full_seq",
                    verbose=True):
    
    def validate_input(input_received, supported_types):
        if input_received not in supported_types:
            raise BaseException("Unsupported device %s, device must be in: %s" %\
                                  (input_received, ", ".join(supported_types)))
                
    
    supported_devices = ["cpu", "mps", "cuda"]    
    validate_input(device, supported_devices)
    
    # Select device

    if device == "cpu":
        device = torch.device("cpu")
    elif device == "mps":    
        device = torch.device("mps" if torch.backends.mps.is_available() and torch.backends.mps.is_built() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() and torch.backends.cuda.is_built() else "cpu")
        
    if verbose:
        print("\t1. Using device: %s" % str(device))
            
    
    # Select model
    model, esm2_alphabet  = load_esm2_model_and_alphabet(model_name)
    model = model.to(device)
    
    if verbose:
        print("\t2. Using esm base model: %s" % model_name)
    
    supported_loss_functions = ["dkl", "orpo", "jsd", "nll", "dpo", "max_diff"]
    
    if type(loss) != list:
        loss = [loss]
        
    loss_to_use = dict([(s, False) for s in supported_loss_functions])
    
    for sub_loss in loss:
        validate_input(sub_loss, supported_loss_functions)                
        loss_to_use[sub_loss] = True
        
    missing_keys = [k for k in loss if k not in loss_weights.keys()]
    if len(missing_keys) > 0:
       raise BaseException("Missing loss weight for loss terms: %s" % ", ".join(missing_keys))
                
    loss_str = "_".join(loss)    
    
    if verbose:
        print("\t3. Using loss: %s" % ", ".join(loss))
        
        
    if verbose:
        print("\t4. Learning rate %.8f, weight decay: %.3f" % (learning_rate, weight_decay))
        
    
    supported_indices_modes = ["mutations", "indices", "random_sample"]
    
    
    if type(indices) != dict and len(indices) != 1:
        raise BaseException("When specifying indices, use the following format: indices = {\"mutations\": RELEVANT PARAMS}")
    
    indices_mode = [k for k in indices.keys()][0]
    
    validate_input(indices_mode, supported_indices_modes)
    
    if indices_mode == "mutations":
        mutations_to_include = indices[indices_mode]
        
        train_mutations_func=lambda sdf: get_indices(sdf, mutations_to_include, nmuts_column=num_muts_column_name),
        test_mutations_func=lambda sdf: get_indices(sdf, mutations_to_include, nmuts_column=num_muts_column_name, rev=True),
        
        muts_str = "muts_%s" % "_".join([str(m) for m in mutations_to_include])
        
        print("\t5. Indices based on mutations: %s" % ", ".join([str(m) for m in mutations_to_include]))
    else:
        raise BaseException("Random sample and ind are currently not supported")
        
        
        
    supported_masks = ["both", "partial", "full"]
    validate_input(mask_type, supported_masks)
    
    if mask_type == "both":
        use_full_only = False
        use_partial_only = False
    elif mask_type == "partial":
        use_full_only = False
        use_partial_only = True
    else:
        use_full_only = True
        use_partial_only = False
    
    
    print("\t6. Mask type chosen: %s" % mask_type)
    
    project_name = "model_%s_loss_%s_indices_%s_mask_type_%s_lr_%.8f_wd_%.3f_iter_%d_bs_%d" %\
        (model_name,
         loss_str,
         muts_str,
         mask_type,
         learning_rate,
         weight_decay,
         iterations,
         batch_size)
        
    
    eval_path = "%s/%s" % (save_path, project_name)
    os.makedirs(eval_path, exist_ok=True)
    weights_path = "%s/weights" % (eval_path)
    os.makedirs(weights_path, exist_ok=True)
    
    
    dataset_class_kwargs = {}
    dataset_class_kwargs["train_project_name"] = project_name
    dataset_class_kwargs["dataset_path"] = dataset_path
    dataset_class_kwargs["evaluation_path"] = eval_path
    dataset_class_kwargs["train_indices"] = train_mutations_func
    dataset_class_kwargs["test_indices"] = test_mutations_func
    dataset_class_kwargs["esm_model"] = model
    dataset_class_kwargs["esm_alphabet"] = esm2_alphabet
    dataset_class_kwargs["full_mask_mut_positions"] = full_mask_pos_func
    dataset_class_kwargs["partial_mask_mut_positions"] = partial_mask_pos_func
    dataset_class_kwargs["use_full_mask_only"] = use_full_only
    dataset_class_kwargs["use_partial_mask_only"] = use_partial_only
    dataset_class_kwargs["ref_seq"] = ref_seq
    dataset_class_kwargs["model_name"] = model_name
    dataset_class_kwargs["sequence_column_name"] = sequence_column_name
    dataset_class_kwargs["activity_column_name"] = activity_column_name
    dataset_class_kwargs["mini_batch_size"] = batch_size
    dataset_class_kwargs["cache"] = True
    dataset_class_kwargs["labels_dtype"] = torch.int64
    
    return(model, 
           esm2_alphabet,
           project_name, 
           loss_to_use,
           weights_path,
           Esm2SequenceActivityTrainTest(**dataset_class_kwargs))

def train_esm_model(dataset_path=None,
                    save_path=None,
                    device="cpu",
                    model_name="esm2_t33_650M_UR50D",
                    pretrained_weights_path=None,
                    loss=["nll", "orpo", "dkl"],
                    loss_weights={'dkl': 1, 'orpo': 1, 'nll': 1},
                    learning_rate=1e-5,
                    weight_decay=0.1,
                    indices=None,
                    batch_size=20,
                    iterations=20000,
                    checkpoint_every=5000,
                    best_sep_checkpoint=20,
                    mask_type="Both",   
                    ref_seq=None,
                    full_mask_pos_func=get_mutated_position_function_gfp,
                    partial_mask_pos_func=get_mutated_position_function_gfp_n2,
                    num_muts_column_name="num_muts",
                    activity_column_name='inactive',
                    sequence_column_name="full_seq",
                    verbose=True):       
    
    

    model, esm2_alphabet, project_name, loss_to_use, weights_path, dataset =\
              validate_args(dataset_path=dataset_path,
                            save_path=save_path,
                            device=device,
                            model_name=model_name,
                            pretrained_weights_path=pretrained_weights_path,
                            loss=loss,
                            loss_weights=loss_weights,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            indices=indices,
                            batch_size=batch_size,
                            iterations=iterations,
                            checkpoint_every=checkpoint_every,
                            best_sep_checkpoint=best_sep_checkpoint,
                            mask_type=mask_type,   
                            ref_seq=ref_seq,
                            full_mask_pos_func=full_mask_pos_func,
                            partial_mask_pos_func=partial_mask_pos_func,
                            num_muts_column_name=num_muts_column_name,
                            activity_column_name=activity_column_name,
                            sequence_column_name=sequence_column_name,
                            verbose=verbose)
    
    len_ref_seq = len(ref_seq)
    

    # batch_size is 1 as batches are internally implemented
    train_data_loader = torch.utils.data.DataLoader(dataset, 
                                                    batch_size=1, 
                                                    shuffle=True)
    
    
    optimizer = torch.optim.Adam(model.parameters(), 
                                 lr=learning_rate, 
                                 weight_decay=weight_decay)
    
    # pretrained_weights_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/retraining_esm/model_esm2_t33_650M_UR50D_loss_nll_dkl_orpo_indices_muts_1_2_3_4_5_mask_type_partial_lr_0.00001000_wd_0.100_iter_20000_bs_20/weights/final_checkpoint_model_esm2_t33_650M_UR50D_loss_nll_dkl_orpo_indices_muts_1_2_3_4_5_mask_type_partial_lr_0.00001000_wd_0.100_iter_20000_bs_20.pt"
    
    # if pretrained_weights_path is not None:
    #     params = torch.load(pretrained_weights_path)
    #     model.load_state_dict(params["model_params "])
        
        
    #     optimizer = torch.optim.Adam(model.parameters(), 
    #                                  lr=learning_rate, 
    #                                  weight_decay=weight_decay)
    #     optimizer.load_state_dict(params["optimizer_params"])
        
    #     iterations -= params["checkpoint"]
    
    loss = torch.nn.CrossEntropyLoss().to(device)
    
    dkl_loss = torch.nn.KLDivLoss().to(device)
    
    dataset.train_dataset_partial_mask.one_hot_mut_info =\
        dataset.train_dataset_partial_mask.one_hot_mut_info.to(device)
        
    dataset.train_dataset_full_mask.one_hot_mut_info =\
        dataset.train_dataset_full_mask.one_hot_mut_info.to(device)      
        
    separation = torch.tensor(0)
            
    wt_tokens = torch.tensor(esm2_alphabet.encode("<cls>" + ref_seq + "<eos>"), dtype=torch.int64, device=device).view((1,-1))
    eos_token = torch.tensor(esm2_alphabet.encode("<eos>"), dtype=torch.int64, device=device) 
    mask_token = torch.tensor(esm2_alphabet.encode("<mask>"), dtype=torch.int64, device=device)            
    
    
    designed_pos = [int(x[1:]) for x in dataset.train_dataset_partial_mask.sequence_dataframe.columns[3:].to_list()]
    designed_pos = torch.tensor(designed_pos)
    plm_positions_to_run = torch.tensor(range(0,240))
    plm_positions_to_run = designed_pos
    run_designed_only=True
    
    ongoing_s_dkl = 0
    ongoing_f_dkl = 0    
    ongoing_s_mxdl = 0
    ongoing_f_mxdl = 0 
    
    n_epochs = ceil(iterations / len(train_data_loader))
    total_steps = 0
    
    if run_designed_only:
        for epoch in range(0, n_epochs):
            for iter_step, batch in enumerate(train_data_loader):
                
                total_steps += 1
                # training loop
                model.train()                    
                
                if mask_type == "both":       
                    positives = batch[0][0].to(device)
                    negatives = batch[0][1].to(device)
                    one_hot_positions = batch[0][2].view((1,1,-1)).to(device)
                    pair_indices = batch[0][3]
                    is_full = batch[1][0] == "full"
                else:                                
                    positives = batch[0].to(device)
                    negatives = batch[1].to(device)
                    one_hot_positions = batch[2].view((1,1,-1)).to(device)
                    pair_indices = batch[3]
                    is_full = mask_type == "full"
                
                _, B, S = positives.size()
                
                pad=torch.tensor(0, device=device).view((1,1,1))
                padded_masked_tensor = torch.cat([pad, one_hot_positions, pad], dim=2)
                padded_mutated_positions = (padded_masked_tensor == 1)
                
        
                batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64, device=device) - padded_masked_tensor) * positives[0,0,:])
                batched_sequences_to_run_with_masks += padded_masked_tensor * mask_token # add masks                
                
                
                optimizer.zero_grad()            
                logits = model(batched_sequences_to_run_with_masks.view((1,-1))[:,plm_positions_to_run])
                masked_logits = logits["logits"][0,:,:]
                        
                
                total_loss = torch.tensor(0, device=device, dtype=torch.float)
                loss_str = []
                
                pssm = masked_logits.softmax(dim=1).view((len(plm_positions_to_run), -1))
                
                sns.heatmap(pssm.cpu().detach().numpy())
                plt.show()
                
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
                        
                                        
                mutated_positions = (one_hot_positions == 1)[:,:,plm_positions_to_run - 1]
                        
                from_mutation = relevant_one_hot[0][indices,:][:,plm_positions_to_run - 1]
                to_mutation = relevant_one_hot[1][indices,:][:,plm_positions_to_run - 1]
                 
                
                predicted_fitness =\
                    fitness_from_prob_non_dms(pssm[mutated_positions.view(-1)].clone().detach(),#pssm[mutated_positions],
                                                  from_mutation[:,mutated_positions.view((-1))][0].clone().detach(),
                                                  to_mutation[:,mutated_positions.view((-1))].clone().detach(),
                                                  device=device)
                        
                actives = predicted_fitness[labels == 1]
                inactives = predicted_fitness[labels == 0]
                        
                
                print("Mx[A]%.3f, Mx[I]%.3f, Mu[A]%.3f, Mu[I]%.3f" %
                      (actives.max().item(),
                      inactives.max().item(),
                      actives.mean().item(),
                      inactives.mean().item()))
                
                if loss_to_use["nll"] or loss_to_use["orpo"] or loss_to_use["dpo"]:                       
                    masked_logits_repeat = masked_logits.repeat(B,1,1)
                    masked_logits_repeat = einops.rearrange(masked_logits_repeat, 'B S C -> B C S')
                    positive_generations = positives[:,:,plm_positions_to_run.view(-1)].squeeze(dim=0)
                    
                if loss_to_use["nll"]:
                    
                    nll_loss = loss(masked_logits_repeat, positive_generations)            
                    loss_str.append("NLL: %.3f" % nll_loss.item())
                    total_loss += torch.tensor(loss_weights["nll"], device=device) * nll_loss
                    
                if loss_to_use["orpo"] or loss_to_use["dpo"]:
                    negative_generations = negatives[:,:,plm_positions_to_run.view(-1)].squeeze(dim=0)                                
                    probs =  masked_logits_repeat.softmax(dim=1)
                            
                    
                    if loss_to_use["orpo"]:                
                        preferred_probs = torch.gather(probs,
                                                       1, 
                                                       positive_generations.unsqueeze(dim=1))
                    
                        unpreferred_probs = torch.gather(probs,
                                                         1, 
                                                         negative_generations.unsqueeze(dim=1))
                        
                        preferred_probs = preferred_probs[:,:,mutated_positions.view(-1)]
                        unpreferred_probs = unpreferred_probs[:,:,mutated_positions.view(-1)]
                        
                        odds_preferred = (preferred_probs.mean(dim=2)) / (1 - preferred_probs.mean(dim=2))
                        odds_unpreferred = (unpreferred_probs.mean(dim=2)) / (1 - unpreferred_probs.mean(dim=2))
                        orpo_loss = -F.logsigmoid((odds_preferred/odds_unpreferred).log()).mean()
                        
                        loss_str.append("ORPO: %.3f" % orpo_loss.item())
                        total_loss += torch.tensor(loss_weights["orpo"], device=device) * orpo_loss            
    
                total_loss.backward()
                print("Loss (%.3f [%s]) [Epoch %d, I %d]" %\
                      (total_loss.item(),
                       " ".join(loss_str),
                       epoch, 
                       iter_step))
                    
                optimizer.step()
                     
                            
                if total_steps % checkpoint_every == 0:
                    
                    model_params = model.state_dict()
                    optimizer_params = optimizer.state_dict()
                    saved_dict = {"model_params ":model_params,
                                  "optimizer_params":optimizer_params,
                                  "checkpoint":total_steps}
                    
                    torch.save(saved_dict, "%s/final_checkpoint_%s.pt" % (weights_path, project_name))
        
        model_params = model.state_dict()
        optimizer_params = optimizer.state_dict()
        saved_dict = {"model_params ":model_params,
                      "optimizer_params":optimizer_params}
        
        torch.save(saved_dict, "%s/final_%d_%s.pt" % (weights_path, iterations, project_name))
    else:
        for epoch in range(0, n_epochs):
            for iter_step, batch in enumerate(train_data_loader):
                
                total_steps += 1
                # training loop
                model.train()                    
                
                if mask_type == "both":       
                    positives = batch[0][0].to(device)
                    negatives = batch[0][1].to(device)
                    one_hot_positions = batch[0][2].view((1,1,-1)).to(device)
                    pair_indices = batch[0][3]
                    is_full = batch[1][0] == "full"
                else:                                
                    positives = batch[0].to(device)
                    negatives = batch[1].to(device)
                    one_hot_positions = batch[2].view((1,1,-1)).to(device)
                    pair_indices = batch[3]
                    is_full = mask_type == "full"
                
                _, B, S = positives.size()
                
                pad=torch.tensor(0, device=device).view((1,1,1))
                padded_masked_tensor = torch.cat([pad, one_hot_positions, pad], dim=2)
                padded_mutated_positions = (padded_masked_tensor == 1)
                
        
                batched_sequences_to_run_with_masks = ((torch.ones(padded_masked_tensor.shape, dtype=torch.int64, device=device) - padded_masked_tensor) * positives[0,0,:])
                batched_sequences_to_run_with_masks += padded_masked_tensor * mask_token # add masks                
                
                
                optimizer.zero_grad()            
                logits = model(batched_sequences_to_run_with_masks.view((1,-1)))
                masked_logits = logits["logits"][0,:,:]
                        
                
                total_loss = torch.tensor(0, device=device, dtype=torch.float)
                loss_str = []
                
                if loss_to_use["dkl"] or loss_to_use["max_diff"]:
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
                        
                    actives = predicted_fitness[labels == 1]
                    inactives = predicted_fitness[labels == 0]
                        
                    
                    
                    if loss_to_use["max_diff"]:
                        mxdl = -(actives.max() - inactives.max())
                        
                        
                        if actives.max().cpu().item() > inactives.max().cpu().item():
                            ongoing_s_mxdl += 1
                        else: 
                            ongoing_f_mxdl += 1
                        
                        print("\t[MAX stats -> S%d:F%d] - %.3f : %.3f" %\
                              (ongoing_s_mxdl, 
                               ongoing_f_mxdl, 
                               actives.max().cpu().item(),
                               inactives.max().cpu().item()))
                        
                        loss_str.append("MAXDL: %.3f" % mxdl.item())
                        total_loss += torch.tensor(loss_weights["max_diff"], device=device) * mxdl
                    
                    if loss_to_use["dkl"]:
                        dkl = -dkl_loss(predicted_fitness,labels.to(torch.float32))
                        
                        if actives.mean().cpu().item() > inactives.mean().cpu().item():
                            ongoing_s_dkl += 1
                        else: 
                            ongoing_f_dkl += 1
                        
                        print("\t[DKL stats -> S%d:F%d] - %.3f : %.3f" %\
                              (ongoing_s_dkl, 
                               ongoing_f_dkl, 
                               actives.mean().cpu().item(),
                               inactives.mean().cpu().item()))
                        
                        
                        loss_str.append("DKL: %.3f" % dkl.item())
                        total_loss += torch.tensor(loss_weights["dkl"], device=device) * dkl
                        
                    
                if loss_to_use["nll"] or loss_to_use["orpo"] or loss_to_use["dpo"]:                
                    masked_logits_repeat = masked_logits[padded_mutated_positions.view((-1)),:].repeat(B,1,1)
                    masked_logits_repeat = einops.rearrange(masked_logits_repeat, 'B S C -> B C S')
                    positive_generations = positives[:,:,padded_mutated_positions.view(-1)].squeeze(dim=0)
                    
                
                if loss_to_use["nll"]:
                    
                    nll_loss = loss(masked_logits_repeat, positive_generations)            
                    loss_str.append("NLL: %.3f" % nll_loss.item())
                    total_loss += torch.tensor(loss_weights["nll"], device=device) * nll_loss
                    
                if loss_to_use["orpo"] or loss_to_use["dpo"]:
                    negative_generations = negatives[:,:,padded_mutated_positions.view(-1)].squeeze(dim=0)            
                    probs =  masked_logits_repeat.softmax(dim=1)
                    
                    
                    #if loss_to_use["orpo"]:
                    # log_probs = probs.log()
        
                    # preferred_log_probs = torch.gather(log_probs,
                    #                                    1, 
                    #                                    positive_generations.unsqueeze(dim=1))
                
                    # unpreferred_log_probs = torch.gather(log_probs,
                    #                                      1, 
                    #                                      negative_generations.unsqueeze(dim=1))
                    
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
                    # dpo_loss = -F.logsigmoid(preferred_log_probs - unpreferred_log_probs).mean()
                    
                    if loss_to_use["orpo"]:                
                        preferred_probs = torch.gather(probs,
                                                       1, 
                                                       positive_generations.unsqueeze(dim=1))
                    
                        unpreferred_probs = torch.gather(probs,
                                                         1, 
                                                         negative_generations.unsqueeze(dim=1))
        
                        odds_preferred = (preferred_probs.mean(dim=2)) / (1 - preferred_probs.mean(dim=2))
                        odds_unpreferred = (unpreferred_probs.mean(dim=2)) / (1 - unpreferred_probs.mean(dim=2))
                        orpo_loss = -F.logsigmoid((odds_preferred/odds_unpreferred).log()).mean()
                        
                        loss_str.append("ORPO: %.3f" % orpo_loss.item())
                        total_loss += torch.tensor(loss_weights["orpo"], device=device) * orpo_loss            

                total_loss.backward()
                print("Loss (%.3f [%s]) [Epoch %d, I %d]" %\
                      (total_loss.item(),
                       " ".join(loss_str),
                       epoch, 
                       iter_step))
                    
                optimizer.step()
                
                if best_sep_checkpoint is not None:
                    if total_steps % best_sep_checkpoint == 0:
                        evaluation_metric =\
                            dataset.evaluate_full(is_msa_transformer=False, 
                                                  return_act_inact=True,
                                                  device=device,
                                                  train=False,
                                                  plot=False)
                        new_separation = np.median(evaluation_metric[0]) - np.median(evaluation_metric[1])
                        
                        if new_separation > separation:
                            print("Saving new model (improved from %.3f to %.3f" % (separation, new_separation))
                            separation = new_separation
                            
                            model_params = model.state_dict()
                            optimizer_params = optimizer.state_dict()
                            saved_dict = {"model_params ":model_params,
                                          "optimizer_params":optimizer_params}
                            
                            torch.save(saved_dict, "%s/best_sep_%s.pt" % (weights_path, project_name))
            
                            dataset.evaluate_full(is_msa_transformer=False, 
                                                  return_act_inact=True,
                                                  device=device,
                                                  train=False,
                                                  plot=True,
                                                  save=True)
                            
                            
                if total_steps % checkpoint_every == 0:
                    
                    model_params = model.state_dict()
                    optimizer_params = optimizer.state_dict()
                    saved_dict = {"model_params ":model_params,
                                  "optimizer_params":optimizer_params,
                                  "checkpoint":total_steps}
                    
                    torch.save(saved_dict, "%s/final_checkpoint_%s.pt" % (weights_path, project_name))
        
        model_params = model.state_dict()
        optimizer_params = optimizer.state_dict()
        saved_dict = {"model_params ":model_params,
                      "optimizer_params":optimizer_params}
        
        torch.save(saved_dict, "%s/final_%d_%s.pt" % (weights_path, iterations, project_name))

def evaluate(dataset_path=None,
                    save_path=None,
                    device="cpu",
                    model_name="esm2_t33_650M_UR50D",
                    pretrained_weights_path=None,
                    loss=["nll", "orpo", "dkl"],
                    loss_weights={'dkl': 1, 'orpo': 1, 'nll': 1},
                    learning_rate=1e-5,
                    weight_decay=0.1,
                    indices=None,
                    batch_size=20,
                    iterations=20000,
                    checkpoint_every=5000,
                    best_sep_checkpoint=20,
                    mask_type="Both",   
                    ref_seq=None,
                    full_mask_pos_func=get_mutated_position_function_gfp,
                    partial_mask_pos_func=get_mutated_position_function_gfp_n2,
                    num_muts_column_name="num_muts",
                    activity_column_name='inactive',
                    sequence_column_name="full_seq",
                    verbose=True):       
    

    model, esm2_alphabet, project_name, loss_to_use, weights_path, dataset =\
              validate_args(dataset_path=dataset_path,
                            save_path=save_path,
                            device=device,
                            model_name=model_name,
                            pretrained_weights_path=pretrained_weights_path,
                            loss=loss,
                            loss_weights=loss_weights,
                            learning_rate=learning_rate,
                            weight_decay=weight_decay,
                            indices=indices,
                            batch_size=batch_size,
                            iterations=iterations,
                            checkpoint_every=checkpoint_every,
                            best_sep_checkpoint=best_sep_checkpoint,
                            mask_type=mask_type,   
                            ref_seq=ref_seq,
                            full_mask_pos_func=full_mask_pos_func,
                            partial_mask_pos_func=partial_mask_pos_func,
                            num_muts_column_name=num_muts_column_name,
                            activity_column_name=activity_column_name,
                            sequence_column_name=activity_column_name,
                            verbose=verbose)
              
    eval_path = "%s/%s" % (save_path, project_name)
    weights_path = "%s/weights" % (eval_path)   
    #params = torch.load("%s/best_sep_%s.pt" % (weights_path, project_name))
    params = torch.load("%s/final_checkpoint_%s.pt" % (weights_path, project_name))
    #"
    model.load_state_dict(params["model_params "])
    
    #     model.load_state_dict(params["model_params "])
        
        
    #     optimizer = torch.optim.Adam(model.parameters(), 
    #                                  lr=learning_rate, 
    #                                  weight_decay=weight_decay)
    #     optimizer.load_state_dict(params["optimizer_params"])
        
    #     iterations -= params["checkpoint"]
    
    dataset.evaluate_across_masks(device=device)   


    
def main():
    parser = argparse.ArgumentParser(description="A simple tool for re-training PLMs for protein function prediction.")
    
    # Add arguments
    parser.add_argument('--yaml_path', type=str, required=False, help='Path to yaml configuration for training')
    parser.add_argument('--mode', type=str, required=False, help='Has to be train or evaluate')

    # Parse arguments
    args = parser.parse_args()

    
    #yaml_path = args.yaml_path
    yaml_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/retraining_esm/example_conf.yaml"
    
        
    with open(yaml_path, "r") as file:
        yaml_args = yaml.safe_load(file)
       
        
    fc = yaml_args.pop("first_mutation_column_name")
    lc = yaml_args.pop("last_mutation_column_name")
    sc = yaml_args["sequence_column_name"]
    
    def full_mask_pos_func(sdf):
        return get_mutated_position_full_mask_by_first_last_colname(sdf, 
                                                                    first_col=fc,
                                                                    last_col=lc,
                                                                    sequence_col=sc)
        
    def partial_mask_pos_func(sdf):
        return get_mutated_position_partial_mask_by_first_last_colname(sdf, 
                                                                       first_col=fc,
                                                                       last_col=lc,
                                                                       sequence_col=sc)
    
    yaml_args["full_mask_pos_func"] = full_mask_pos_func
    yaml_args["partial_mask_pos_func"] = partial_mask_pos_func
    
    
    
    if type(yaml_args["learning_rate"]) != float:
        yaml_args["learning_rate"] = float(yaml_args["learning_rate"])
    
    train_esm_model(**yaml_args)
    
    
if __name__ == "__main__":
        main()
        