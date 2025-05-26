import torch
import torch.nn.functional as F

from constants import *
from utils import *

fix_esm_path()

import esm

import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F

from rosetta_former.energy_vqvae import *
from dataset import *

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Hyperparameters
ENERGY_INPUT_DIM = 20
ENERGY_D_OUT = 512
ENERGY_D_MODEL = 1024
ENERGY_N_CODEBOOK = 16384# 8192
ENCODER_DEPTH=4
DECODER_DEPTH=16

commitment_cost = .5


batch_size = 32
num_epochs = 10


class Metric:
  def __init__(self):
    self.lst = 0.
    self.sum = 0.
    self.cnt = 0
    self.avg = 0.

  def update(self, val, cnt=1):
    self.lst = val
    self.sum += val * cnt
    self.cnt += cnt
    self.avg = self.sum / self.cnt



def accuracy(pred, target):
  #acc = sum(pred == target) / pred.shape[0]
  #acc = torch.sum(pred == target, axis=1) / pred.shape[1]
  #return acc.to(pred).mean()
  acc = (pred.argmax(dim=1) == target)
  return acc.to(pred).mean()




class avg_pool_mlp(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        hidden_features = in_features * 4
        self.norm1 = nn.LayerNorm(in_features)
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act1 = nn.GELU()
        # self.fc2 = nn.Linear(hidden_features, hidden_features)
        # self.act2 = nn.GELU()
        # self.fc3 = nn.Linear(hidden_features, hidden_features)
        # self.act3 = nn.GELU()
        # self.fc4 = nn.Linear(hidden_features, hidden_features)
        # self.act4 = nn.GELU()
        
        self.fc_final = nn.Linear(hidden_features, out_features)
        #self.act_final = nn.Sigmoid()

    def forward(self, x):
        

        x = x.mean(axis=1)        
        x = self.norm1(x)
        x = self.fc1(x)
        x = self.act1(x)
        # x = self.fc2(x)
        # x = self.act2(x)
        # x = self.fc3(x)
        # x = self.act3(x)
        # x = self.fc4(x)
        # x = self.act4(x)
        x = self.fc_final(x)
        #x = self.act_final(x)
        
        return x.softmax(dim=1)


def train(model, optimizer, criterion, train_data_loader):
    loss_metric = Metric()
    acc_metric = Metric()
    
    # Training loop
    for epoch in range(num_epochs):
        ctr = 0
        
#        save_torch_model(model_name, model, optimizer)
        
        for idx, item in enumerate(train_data_loader):
            
            x,y = item
            ctr += 1
            optimizer.zero_grad()

            # Forward pass
            y_pred =  model(x)
            y = y.to(y_pred)
            loss = criterion(y_pred, y.argmax(dim=1))            
            acc = accuracy(y_pred, y.argmax(dim=1))
            
            loss.backward()
            optimizer.step()
            
            loss_metric.update(loss.item(), x.size(0))
            acc_metric.update(acc.item(), x.size(0))
            
            if idx % 20 == 0:
                print("\t (%d). Acc : %.3f, Loss: %.3f" % (idx, acc, loss))
                sns.heatmap(y_pred.detach().numpy())
    

        print(' Train', f'Epoch: {epoch:03d} / {num_epochs:03d}',
            f'Loss: {loss_metric.avg:7.4g}',
            f'Accuracy: {acc_metric.avg:.3f}',
            sep='   ')
        
      

  #  save_torch_model(model_name, model, optimizer)

    print("Training completed!")


base_dataset_path = "/Users/itayta/Desktop/prot_stuff/fitness_lndscp/fitness_learning/data/datasets/random_100k_train"

dataset = EsmGfpDataset(sequences_path="%s/sequences.csv" % base_dataset_path,
                        embeddings_path="%s/embeddings" % base_dataset_path,
                        embedding_layer = -1,
                        tokens_path="%s/tokens" % base_dataset_path,
                        mode="embeddings")

x,y = dataset[0]

model = avg_pool_mlp(x.shape[1], y.shape[0])
optimizer = torch.optim.Adam(model.parameters(), lr=3e-5)
train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
#criterion = nn.BCEWithLogitsLoss(torch.ones([y.shape[0]]))#F.cross_entropy
criterion = F.cross_entropy
train(model, optimizer, criterion, train_data_loader)