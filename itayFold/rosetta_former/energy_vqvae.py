# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F

class GumbelQuantize(nn.Module):
    """
    Gumbel Softmax trick quantizer
    Categorical Reparameterization with Gumbel-Softmax, Jang et al. 2016
    https://arxiv.org/abs/1611.01144
    """
    def __init__(self, num_embeddings, embedding_dim, straight_through=False):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_embed = num_embeddings

        self.straight_through = straight_through
        self.temperature = 1.0
        self.kld_scale = 5e-5

        self.num_hiddens=embedding_dim
        self.proj = nn.Conv1d(self.num_hiddens, num_embeddings, 1)
        self.embed = nn.Embedding(num_embeddings, embedding_dim)
        

    def forward(self, z):

        # force hard = True when we are in eval mode, as we must quantize
        hard = self.straight_through if self.training else True

        logits = self.proj(torch.transpose(z.to(torch.float), 1, 2))
        soft_one_hot = F.gumbel_softmax(logits, tau=self.temperature, dim=1, hard=hard)
        z_q = torch.einsum('b n s, n d -> b d s', soft_one_hot, self.embed.weight).transpose(1, 2)
        
        # + kl divergence to the prior loss
        qy = F.softmax(logits, dim=1)
        diff = self.kld_scale * torch.sum(qy * torch.log(qy * self.n_embed + 1e-10), dim=1).mean()

        ind = soft_one_hot.argmax(dim=1)
        return z_q, diff, ind
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, dtype=torch.double):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        # Initialize embeddings
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim, dtype=dtype)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, inputs):
        # Flatten the input
        B, L, H = inputs.size()
        flat_inputs = inputs.view(-1, self.embedding_dim)

        # Compute distances between input and embedding vectors
        distances = torch.sum(flat_inputs ** 2, dim=1, keepdim=True) \
                    + torch.sum(self.embeddings.weight ** 2, dim=1) \
                    - 2 * torch.matmul(flat_inputs, self.embeddings.weight.t())

        # Find the closest embedding for each input
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embeddings(encoding_indices).view_as(inputs)

        # Compute the VQ-VAE loss
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Add the straight-through estimator for gradients
        quantized = inputs + (quantized - inputs).detach()
        encoding_indices = encoding_indices.view(B, L, -1)
        
        return quantized, loss, encoding_indices
    

    
class EncoderBlockEnergyVQVAE(nn.Module):
    def __init__(self, input_dim, out_dim, dtype=torch.double):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, out_dim, dtype=dtype),
            nn.ReLU(),
            nn.LayerNorm(out_dim, dtype=dtype),
            #nn.Linear(out_dim, out_dim, dtype=dtype)      
        )
        
    def forward(self, x):    
        x = self.encoder(x)        
        return(x)        
    
class  EncoderEnergyVQVAE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 d_model, 
                 d_out, 
                 num_embeddings, 
                 commitment_cost, 
                 n_blocks=1,
                 dtype=torch.double):
        super().__init__()
        
        self.encoding_blocks = nn.ModuleList([
            EncoderBlockEnergyVQVAE(input_dim if i == 0 else d_model, 
                                    d_model)
                
            for i in range(0, n_blocks)])
        
        self.vq_proj_layer = nn.Linear(d_model, d_out, dtype=dtype)
        #self.vq_layer = VectorQuantizer(num_embeddings, d_out, commitment_cost)
        self.vq_layer = GumbelQuantize(num_embeddings, d_out)
        
    
    def forward(self, x):
        for block in self.encoding_blocks:
            x = block(x)
        
        
        x = self.vq_proj_layer(x)
        z_q, vq_loss, encoding_ind = self.vq_layer(x)
        
        return z_q, vq_loss, encoding_ind


class DecoderBlockEnergyVQVAE(nn.Module):
    def __init__(self, d_model, output_dim, is_final_layer=False, dtype=torch.double):
        super().__init__()
        
        if is_final_layer:
            self.decoder = nn.Sequential(
                nn.Linear(d_model, output_dim, dtype=dtype),
                nn.ReLU(),
                nn.LayerNorm(output_dim, dtype=dtype),
                nn.Linear(output_dim, output_dim, dtype=dtype),      
                #nn.Sigmoid() This fucked me during training lol
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(d_model, output_dim, dtype=dtype),
                nn.ReLU(),
                nn.LayerNorm(output_dim, dtype=dtype),
                nn.Linear(output_dim, output_dim, dtype=dtype)
                
            )
            
    def forward(self, x):    
        x = self.decoder(x)        
        return(x)
                

class DecoderEnergyVQVAE(nn.Module):
    def __init__(self, 
                 d_model, 
                 output_dim, 
                 d_out, 
                 num_embeddings, 
                 n_blocks=1, 
                 use_raw_tokens=True,
                 dtype=torch.double):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, d_model,  dtype=dtype)
        self.vq_proj_layer = nn.Linear(d_out, d_model,  dtype=dtype)
        self.use_raw_tokens = use_raw_tokens
        self.decoding_blocks = nn.ModuleList([
            DecoderBlockEnergyVQVAE(d_model, 
                                    output_dim if i == (n_blocks - 1) else d_model,
                                    True if i == (n_blocks - 1) else False)
                
            for i in range(0, n_blocks)])
        
    def forward(self, x, use_tokens=True):
        
        if self.use_raw_tokens:
            x = self.embedding(x)
            x =  x.squeeze()
        else:
            x = self.vq_proj_layer(x.to(torch.double))
            
        for block in self.decoding_blocks:
            x = block(x)
        
        return x

class EnergyVQVAE(nn.Module):
    def __init__(self, 
                 input_dim, 
                 d_model,
                 d_out,
                 num_embeddings, 
                 commitment_cost,
                 encoder_depth=2,
                 decoder_depth=2, 
                 use_raw_tokens=False,
                 dtype=torch.double):
        super().__init__()        
        self.use_raw_tokens = use_raw_tokens
        self.encoder_net = EncoderEnergyVQVAE(input_dim=input_dim, 
                                              d_model=d_model, 
                                              d_out=d_out, 
                                              num_embeddings=num_embeddings,
                                              commitment_cost=commitment_cost, 
                                              n_blocks=encoder_depth)
        
        self.decoder_net = DecoderEnergyVQVAE(d_model=d_model, 
                                              output_dim=input_dim, 
                                              d_out=d_out, 
                                              num_embeddings=num_embeddings,
                                              n_blocks=decoder_depth,
                                              use_raw_tokens=use_raw_tokens)
    
        
    def forward(self, x):
        # Encode
        z_q, vq_loss, encoding_ind = self.encoder_net(x)

        # Decode
        if self.use_raw_tokens:
            x_recon = self.decoder_net(encoding_ind)
        else:
            x_recon = self.decoder_net(z_q)

        return x_recon, vq_loss, encoding_ind




