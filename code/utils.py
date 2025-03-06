#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 16:24:57 2025

@author: itayta
"""

import sys, os
import pandas as pd
import numpy as np
import csv

from datetime import datetime
from constants import *

original_sys_path = ""


class DesignConfigurationMeta(type):
    """A metaclass for creating Singleton classes."""
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            # Create a new instance if it doesn't exist
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class DesignConfiguration(metaclass=DesignConfigurationMeta):
    
    is_fixed=True
    
    if is_fixed:
        sequence_space_path = FIXED_SEQUENCE_SPACE_FILE_PATH
    else:
        sequence_space_path = SEQUENCE_SPACE_FILE_PATH
    
    def __init__(self,):
        self.df = pd.read_csv(self.sequence_space_path)

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

def file_in_dir(file_name, directory):
    for root, dirs, files in os.walk(directory):
        if file_name in files:
            return True
    return False
        

def get_pdb_path_from_idx(idx, is_sequence_indices, full_path=True):        
    sequence_df = DesignConfiguration().df    
    uid_column = "sequence" if is_sequence_indices else "idx"
    seq_job_df = sequence_df[sequence_df[uid_column] == idx]
    seq_str = seq_job_df["sequence"].iloc[0]
    pdb_file_name = "%s_refined.pdb" %  seq_str
    
    if full_path:
        pdb_file_name = "%s/%s" % PDB_ROSETTA_SCORES_PATH
    
    return(pdb_file_name)


def parse_energys_matrix_from_pdb(pdb_path):
    a = open(pdb_path, "r")
    lns = a.readlines()
    a.close()
    start_index = next((i for i, s in enumerate(lns) if s.startswith("label")), None)
    end_index = next((i for i, s in enumerate(lns) if s.startswith("#END")), None)
    
    raw_energies_str = [(s[0:-1]).split(" ") for s in lns[start_index:end_index]]
    energies = [[float(x) for x in energy_str[1:-1]] for energy_str in raw_energies_str[2:]]
    energies_mat = np.vstack(energies)
    labels = raw_energies_str[0][1:-1]

    return(energies_mat, labels)


def gfp_rosetta_job_generator(indices, is_sequence_indices, gen_pdb, override=False):
    jobs = []
    if not is_sequence_indices:
        indices = [int(i) for i in indices]
    print(indices)

    uid_column = "sequence" if is_sequence_indices else "idx"
    
    sequence_df = DesignConfiguration().df    
    aa_columns = sequence_df.columns[-N_POSITIONS_IN_DESIGN_SPACE:]
    
    all_poss = ["%sA" % p[1:] for p in aa_columns.to_list()]
    all_residues = ",".join(all_poss)

    hbonds_all_residues = [14, 16, 18, 42, 44, 46, 61, 64, 68, 69, 72, 108, 110, 112, 119, 123, 145, 150, 163, 165, 167, 181, 185, 201, 220, 224]    
    nohbonds_all_residues = [42, 44, 61, 62, 69, 92, 94, 96, 112, 121, 145, 148, 150, 163, 165, 167, 181, 183, 185, 203, 205, 220, 222, 224]

    hbonds_all_residues = ",".join(["%dA" % aa for aa in hbonds_all_residues])
    nohbonds_all_residues = ",".join(["%dA" % aa for aa in nohbonds_all_residues])

    pdb_files = []
    
    for ind in indices:
    
        seq_job_df = sequence_df[sequence_df[uid_column] == ind]

        seq_str = seq_job_df["sequence"].iloc[0]
        pdb_file_name = "%s_refined.pdb" %  seq_str

        # In case already exists and we're not overriding
        if not override and file_in_dir(pdb_file_name, PDB_ROSETTA_SCORES_PATH):
            if VERBOSE:
                print("Found %s in %s, no need to run anyhthing" % (pdb_file_name, PDB_ROSETTA_SCORES_PATH))
                
            pdb_files.append([seq_str, "%s/%s" % (PDB_ROSETTA_SCORES_PATH, pdb_file_name)])
            continue

        aa = seq_job_df[aa_columns].iloc[0,:].to_list()        
        muts = [(v[0], v[1:], i)  for i,v in enumerate(aa_columns.to_list()) if v[0] != aa[i]]
        assert len(muts) == seq_job_df["num_of_muts"].iloc[0]        
        
        mutated_residues = " ".join(["new_res%d=%s target%d=%sA" % (i+1, ONE_2_THREE[v[0]], i+1, v[1]) for i,v in enumerate(muts)])
        
        job = ""
        job += "%s " % ROSETTA_SCRIPTS
        job += "-database %s " % ROSETTA_DB
        job += "@%s/initial_data/flags " % ROSETTA_FILES_PATH
        job += "-out:prefix %s/%s_ " % (PDB_ROSETTA_SCORES_PATH, seq_str)
        job += "-out:file:%s %s/%s.sc " % ("scorefile" if gen_pdb else "score_only", ROSETTA_SCORES_PATH, seq_str)
        job += "-parser:script_vars "
        #job += "all_ress=%s " % all_residues
        if seq_job_df["is_hbonds"].iloc[0] == 1:
            job += "all_ress=%s " % hbonds_all_residues
        if seq_job_df["is_nohbonds"].iloc[0] == 1:
            job += "all_ress=%s " % nohbonds_all_residues
        else: 
            job += "all_ress=%s " % ",".join(sorted(list(set(nohbonds_all_residues.split(",") + hbonds_all_residues.split(","))), key=lambda v: int(v[:-1])))

        job += "%s" % mutated_residues
        
        print(job)
        jobs.append(job)
        #

        if gen_pdb:
            pdb_files.append([seq_str, "%s/%s" % (PDB_ROSETTA_SCORES_PATH, pdb_file_name)])
            gunzip_job = "gunzip %s/%s.gz" % (PDB_ROSETTA_SCORES_PATH, pdb_file_name)
            print(gunzip_job)
            jobs.append(gunzip_job)
    
    return(jobs, pdb_files)
   


def get_missing_sequences(is_fulL_path, seq_space_file, is_test=False, return_full_path=False):
    if not is_fulL_path:
        full_seq_space_file = "%s/%s" % (TRAIN_TEST_SPLITS_PATH, seq_space_file)

    if is_test:
        working_df = pd.read_csv("%s_test.csv" % full_seq_space_file)
    else:
        working_df = pd.read_csv("%s_train.csv" % full_seq_space_file)

    n_sequences = working_df.shape[0]

    sequences_in_set = working_df["sequence"].to_list()
    pdb_generated = os.listdir(PDB_ROSETTA_SCORES_PATH)
    tokens_generated = os.listdir(RAW_TENSORS_PATH)
    energies_generated = os.listdir(RAW_TENSORS_ENERGIES_PATH)



    all_energy_files = ["energies_%s.pth" % seq for seq in sequences_in_set]
    all_pdb_files  =  ["%s_refined.pdb" % seq for seq in sequences_in_set]
    all_esm_tensor_files  = ["tokens_%s.pth" % seq for seq in sequences_in_set]


    if return_full_path:
        pdb_generated = ["%s/%s" % (PDB_ROSETTA_SCORES_PATH, f) for f in  pdb_generated]
        tokens_generated = ["%s/%s" % (RAW_TENSORS_PATH, f) for f in  tokens_generated]
        energies_generated = ["%s/%s" % (RAW_TENSORS_ENERGIES_PATH, f) for f in  energies_generated]
    
        all_pdb_files = ["%s/%s" % (PDB_ROSETTA_SCORES_PATH, f) for f in  all_pdb_files]
        all_esm_tensor_files = ["%s/%s" % (RAW_TENSORS_PATH, f) for f in  all_esm_tensor_files]
        all_energy_files = ["%s/%s" % (RAW_TENSORS_ENERGIES_PATH, f) for f in  all_energy_files]


    results = {"all_energy_files" : all_energy_files,
                "all_pdb_files" : all_pdb_files,
                "all_esm_tensor_files" : all_esm_tensor_files,
                "sequences_in_set" : sequences_in_set,
                "pdb_generated" : pdb_generated,
                "tokens_generated" : tokens_generated,
                "energies_generated" : energies_generated}
                    
    return (results) 

def append_to_csv(file_name, data, headers=None):

    file_exists = os.path.exists(file_name)

    with open(file_name, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.DictWriter(file, fieldnames=headers if headers else data[0].keys())
            
        if not file_exists:
            if headers:
                writer.writeheader()
            else:
                raise ValueError("Headers must be provided when creating a new CSV file.")

        for row in data:
            writer.writerow(row)

def append_to_log(sequence, free_text_1=None, free_text_2=None, ref_file_1=None, ref_file_2=None, log_path=None):
    if log_path is None:
        log_path = DEFAULT_LOG_FILE_PATH

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y-%m-%d %H:%M:%S")
    log_row = {'time_stamp': formatted_time,
               'sequence': sequence,
               'free_text_1' : free_text_1 if free_text_1 is not None else "",
               'free_text_2' : free_text_2 if free_text_2 is not None else "",
               'ref_file_1' : ref_file_1 if ref_file_1 is not None else "",
               'ref_file_2' : ref_file_2 if ref_file_2 is not None else ""}

    headers = ["time_stamp",
               "sequence",
               "free_text_1",
               "free_text_2",
               "ref_file_1",
               "ref_file_2"]

    append_to_csv(log_path, log_row, headers)



    