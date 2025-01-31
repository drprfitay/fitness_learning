#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 12 14:14:18 2025

@author: itayta
"""



VERBOSE = True

ROOT_PATH = "/home/labs/fleishman/itayta/fitness_learning"

DATA_PATH = "%s/data/" % ROOT_PATH
CODE_PATH = "%s/code/" % ROOT_PATH
LOG_PATH = "%s/log/" % ROOT_PATH

CLUSTER_EXECUTIONS_PATH = "%s/executions" % ROOT_PATH
CONFIGURATION_PATH = "%s/configuration" % DATA_PATH
PDB_ROSETTA_SCORES_PATH = "%s/pdb_rosetta_scores" % DATA_PATH
ROSETTA_SCORES_PATH = "%s/rosetta_scores" % DATA_PATH
RAW_TENSORS_PATH = "%s/raw_tensors" % DATA_PATH
RAW_TENSORS_ENERGIES_PATH = "%s/energies" % RAW_TENSORS_PATH
TRAIN_TEST_SPLITS_PATH = "%s/train_test_splits/" % CONFIGURATION_PATH
SEQ_SPACES_PATH = "%s/sequence_spaces/" % CONFIGURATION_PATH
ZIP_PATH = "%s/zip/" % DATA_PATH 

DEFAULT_LOG_FILE_PATH =  "%s/%s" % (LOG_PATH, "default_log.csv")
SEQUENCE_SPACE_FILE_PATH = "%s/%s" % (CONFIGURATION_PATH, "gfp_sequence_dataset.csv")


SEQ_DF_ACTIVITY_LABEL_START_IDX = 6
SEQ_DF_ACTIVITY_LABEL_END_IDX = 39

ROSETTA_SCRIPTS = "/home/labs/fleishman/rosaliel/Rosetta/main/source/build/src/release/linux/3.10/64/x86/gcc/5.4/default/rosetta_scripts.default.linuxgccrelease"
ROSETTA_DB = "/home/labs/fleishman/rosaliel/Rosetta/main/database"

ROSETTA_FILES_PATH = "%s/rosetta" % ROOT_PATH
WT_PDB_PATH = "%s/initial_data/refined.pdb" %  ROSETTA_FILES_PATH

ITAYFOLD_PATH = "%s/itayFold/" % ROOT_PATH

WEIGHTS_PATH = "/%s/weights/" % ITAYFOLD_PATH
MODEL_WEIGHTS_FILE_NAME = "model_weights.pth"
LORA_WEIGHTS_FIlE_NAME =  "lora_weights.pth"
ENCODER_WEIGHTS_FILE_NAME = "structure_encoder.pth"
DECODER_WEIGHTS_FILE_NAME = "structure_decoder.pth"

USE_HF = False
HF_TOKEN = "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"

ENCODER_D_MODEL=1024
ENCODER_N_HEADS=1
ENCODER_V_HEADS=128
ENCODER_N_LAYERS=2
ENCODER_D_OUT=128
ENCODER_N_CODES=4096

DECODER_D_MODEL=1280
DECODER_N_HEADS=20
DECODER_N_LAYERS=30

D_MODEL = 1536
N_LAYERS = 48
N_HEADS = 24
V_HEADS = 256
LORA_R = 16

BOS_STRUCTURE_TOKEN = 4098
EOS_STRUCTURE_TOKEN = 4097 

LORA_TRANSFORMER_LINEAR_WEIGHTS = True
LORA_OUTPUT_HEADS = True

N_POSITIONS_IN_DESIGN_SPACE = 22

# load basic parameters, this requires no changes
AAs = list("ACDEFGHIKLMNPQRSTVWY")

BB_ATOMS = ["N", "CA", "C", "O"]

ONE_2_THREE = {
    "A": "ALA",
    "C": "CYS",
    "D": "ASP",
    "E": "GLU",
    "F": "PHE",
    "G": "GLY",
    "H": "HIS",
    "I": "ILE",
    "K": "LYS",
    "L": "LEU",
    "M": "MET",
    "N": "ASN",
    "P": "PRO",
    "Q": "GLN",
    "R": "ARG",
    "S": "SER",
    "T": "THR",
    "V": "VAL",
    "W": "TRP",
    "Y": "TYR",
}

THREE_2_ONE = {v: k for k, v in ONE_2_THREE.items()}



#### LSF STUFF 
SHORT_Q = "short"
LONG_Q = "long"

MEM="48GB"

MAIN_JOB_MASTER_FILE_NAME = "main.py"

BSUB_COMMAND = "bsub -q %s -R rusage[mem=%s] -C 1 -G fleishman-wx-grp-lsf -o %s -e %s.err -u /dev/null %s"

