
import torch
import torch.nn.functional as F
import loralib as lora
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import Subset
from torch.cpu import device_count
from torch.utils.data import Dataset

class PREDataset(Dataset):
    def __init__(self,
                 dataset_path,
                 indices,
                 encoding_function,
                 encoding_identifier,
                 external_encoding=None,
                 cache=True,
                 sequence_column_name='full_seq',                 
                 label_column_name='activity',
                 pad_region_label=None,
                 labels_dtype=torch.float64):
            
            
        if not dataset_path.endswith(".csv"):
            raise BaseException("Dataset must be a .csv file received (%s)" % dataset_path)

        
        self.dataset_path = dataset_path    
        self.sequence_dataframe = pd.read_csv(dataset_path)            
        self.size = self.sequence_dataframe.shape[0] # ToDo: read from dataset        
        
        self.sequence_column_name=sequence_column_name
        self.label_column_name=label_column_name

        if self.label_column_name is not None and self.label_column_name in self.sequence_dataframe.columns:        
            self.labels = torch.tensor(self.sequence_dataframe[self.label_column_name], dtype=labels_dtype)
        else: 
            self.labels = torch.tensor(np.arange(0, self.size), dtype=torch.int32)

        if pad_region_label is not None and pad_region_label in self.sequence_dataframe.columns:
            self.pad_region = self.sequence_dataframe[pad_region_label].to_numpy() # torch doesn't support strings
        else:
            self.pad_region = None
                    
        self.encoding_function = encoding_function
        self.cache_path = "%s_cache/" % dataset_path.split(".csv")[0]       
        self.cache = cache
    
        
        if external_encoding is None:
        
            os.makedirs(self.cache_path, exist_ok=True)
            os.makedirs("%s/misc" % self.cache_path, exist_ok=True)   
            
            tokenized_sequences_filename = "%s_encoded_sequences.pt"  % encoding_identifier
            cached_files = os.listdir("%s/misc" % self.cache_path)
            
            
            if self.cache and tokenized_sequences_filename in cached_files:
                print("[INFO] Loading cached file %s" % tokenized_sequences_filename)
                self.encoded_tensor = torch.load("%s/misc/%s" % (self.cache_path, tokenized_sequences_filename))
            else:
                print("[INFO] Tokenizing sequences, this may take a while")                              
                encoded_sequences = [torch.tensor(self.encoding_function(seq)) for seq in self.sequence_dataframe[self.sequence_column_name].to_list()]
                self.encoded_tensor = torch.stack(encoded_sequences, dim=0)
                
                if self.cache:
                    torch.save(self.encoded_tensor, "%s/misc/%s" % \
                            (self.cache_path, tokenized_sequences_filename))
                    
                    
        else:
            self.encoded_tensor = external_encoding
        
        # subset based on indices
        if indices is not None:                
            # Monkey patch!!!!!!
            if type(indices) == tuple:
                indices = indices[0]
                        
            #subset based on indices                            
            if callable(indices):
                indices = indices(self.sequence_dataframe)
                        
            if type(indices) == list:
                self.indices = indices                    
            else:
                self.indices = None                    
                
        if indices is None:
            self.indices = [i for i in range(0, self.sequence_dataframe.shape[0])]
        
        indices_tensor = torch.tensor(self.indices)#torch.tensor(sample(self.indices, 100)) ##       #torch.tensor(sample(self.indices, 1000)) ##       #
        # indices_tensor = sample(indices_tensor[self.labels[indices_tensor] == 0].tolist(), torch.unique(self.labels[indices_tensor], return_counts=True)[1][1].int()
        # ) + indices_tensor[self.labels[indices_tensor] == 1].tolist()

        self.encoded_tensor = self.encoded_tensor[indices_tensor,:]            
        self.labels = self.labels[indices_tensor]

        # pad region is a string thus cannot be saved as a tensor
        if self.pad_region is not None:
            self.pad_region = self.pad_region[indices_tensor.to_numpy().astype(int)]
        
        self.size = self.labels.shape[0]

    def __getitem__(self,idx):
        return self.encoded_tensor[idx], self.labels[idx]
                
    def __len__(self):
        return self.size

    def get_pad_regions(self):
        return self.pad_region
        

class PREActivityDataset(Dataset):
        def __init__(self,
                     evaluation_path,
                     dataset_path,
                     train_indices,
                     test_indices,
                     encoding_function,
                     encoding_identifier,  
                     external_encoding=None,
                     cache=True,
                     lazy_load=True,
                     sequence_column_name='full_seq',
                     label_column_name='inactive',
                     ref_seq="",
                     mini_batch_size=20,
                     positive_label=0,
                     negative_label=1,
                     device=torch.device("cpu"),
                     labels_dtype=torch.float64):
            
            self.train_indices=train_indices,
            self.test_indices=test_indices,

            self.evaluation_path=evaluation_path
            self.dataset_path=dataset_path         
            
            self.encoding_function = encoding_function
            self.encoding_identifier = encoding_identifier
            self.external_encoding = external_encoding
            self.cache=cache
            self.lazy_load=lazy_load
            self.sequence_column_name=sequence_column_name
            self.label_column_name=label_column_name
            self.ref_seq=ref_seq
            self.positive_label=positive_label
            self.negative_label=negative_label
            self.labels_dtype=labels_dtype
            self.device = device
        
            if type(self.train_indices) == tuple:
                self.train_indices = self.train_indices[0]
    
            if type(self.test_indices) == tuple:
                self.test_indices = self.test_indices[0]

            if type(dataset_path) == tuple and len(dataset_path) == 2:
                self.train_dataset_path = dataset_path[0]
                self.test_dataset_path = dataset_path[1]
                
            self.train_dataset_path = self.dataset_path
            self.test_dataset_path = self.dataset_path        
            
            self.train_dataset =\
                    PREDataset(self.train_dataset_path,
                                   self.train_indices,
                                   self.encoding_function,
                                   self.encoding_identifier,
                                   self.external_encoding,
                                   self.cache,
                                   self.sequence_column_name,
                                   self.label_column_name,
                                   self.labels_dtype)
                    
            self.size = len(self.train_dataset)

            self.loaded = False

            if not self.lazy_load:
                self.loaded = True
                self.lazy_load_func()
            
        def lazy_load_func(self):

                if not self.loaded:
                    self.test_dataset =\
                        PREDataset(self.test_dataset_path,
                                    self.test_indices,
                                    self.encoding_function,
                                    self.encoding_identifier,
                                    self.external_encoding,
                                    self.cache,
                                    self.sequence_column_name,
                                    self.label_column_name,
                                    self.labels_dtype)
                    self.loaded = True
                    
        def __getitem__(self,idx):
            return self.train_dataset[idx]
                
        def __len__(self):
            return self.size  
            
        @torch.no_grad()
        def evaluate(self,
                     model,
                     eval_func,
                     finalize_func,
                     eval_train=True,
                     eval_test=True,
                     internal_batch_size=20, 
                     **kwargs):

            model = model.to(self.device)
            model.eval()

            confs = []
            if eval_train:
                confs.append({"train": True,
                              "path_prefix": "train"})

            if eval_test:
                confs.append({"train": False,
                              "path_prefix": "test"})
                              #"subsample": "subsample.csv"})            

            for conf in confs:
                is_train = conf["train"]
                path_prefix = conf["path_prefix"]

                if is_train:
                    print("Evaluating train dataset")
                    working_dataset = self.train_dataset
                else:
                    print("Evaluating test dataset")
                    self.lazy_load_func()
                    working_dataset = self.test_dataset
                
                indices_to_save = torch.tensor(working_dataset.indices)

                if "subsample" in conf:
                    
                    subsample_path = conf["subsample"]
                    subsample_df = pd.read_csv(subsample_path)

                    if "indices" in subsample_df.columns:
                        print(f"Sampling dataset from size {len(working_dataset)} to size {len(subsample_df['indices'])}")
                        subsample_indices = subsample_df["indices"].tolist()
                        indices_to_save = torch.tensor(working_dataset.indices)[subsample_indices] # original df indices - subset based on subsample indices
                        working_dataset = Subset(working_dataset, subsample_indices)
                    else:
                        print(f"Warning: 'indices' column not found in {subsample_path}")
                            
                dataloader =\
                     torch.utils.data.DataLoader(working_dataset, batch_size=internal_batch_size, shuffle=False)

                aggregated_evaluated_data = {}
                
                num_sequences = len(working_dataset)
                print(f"[INFO] Evaluating on device: {self.device}")
                print(f"[INFO] About to evaluate {num_sequences} sequences on {'train' if is_train else 'test'} set.")                
                
                for idx, data in enumerate(dataloader):                
                    if idx % ((len(dataloader) // 20) + 1) == 0:
                        print(f"[INFO] Progress: {idx}/{len(dataloader)} batches evaluated")
                        
                    aggregated_evaluated_data = eval_func(model, 
                                                          data, 
                                                          aggregated_evaluated_data, 
                                                          self.device, 
                                                          batch_idx=idx,
                                                          dataset=working_dataset,
                                                          **kwargs)
                    
                evaluated_data_to_save = finalize_func(aggregated_evaluated_data, working_dataset, **kwargs)
                evaluated_data_to_save["indices"] = indices_to_save

                prefix_folder = os.path.join(self.evaluation_path, path_prefix)
                os.makedirs(prefix_folder, exist_ok=True)

                for key, value in evaluated_data_to_save.items():
                    filename = f"{key}"

                    print(f"\t\tSaving {key} to {os.path.join(prefix_folder, filename)} (type: {type(value)})")

                    # Create the prefix folder within evaluation_path                    
                    # Save matplotlib figure
                    if hasattr(value, "savefig"):
                        fig_path = os.path.join(prefix_folder, f"{filename}.png")
                        value.savefig(fig_path)
                        fig_path_pdf = os.path.join(prefix_folder, f"{filename}.pdf")
                        value.savefig(fig_path_pdf, format="pdf")
                        plt.close(value)
                    # Save pandas DataFrame
                    elif hasattr(value, "to_csv"):
                        csv_path = os.path.join(prefix_folder, f"{filename}.csv")
                        value.to_csv(csv_path, index=False)
                    # Save torch tensor
                    elif hasattr(value, "cpu") and hasattr(value, "detach"):
                        tensor_path = os.path.join(prefix_folder, f"{filename}.pt")
                        torch.save(value.cpu().detach(), tensor_path)
                    # Save numpy array
                    elif hasattr(value, "shape") and hasattr(value, "dtype"):
                        npy_path = os.path.join(prefix_folder, f"{filename}.npy")
                        np.save(npy_path, value)
                    else:
                        # Optionally, skip or print warning for unknown types
                        print(f"Warning: Could not save {key}, unknown type {type(value)}")
