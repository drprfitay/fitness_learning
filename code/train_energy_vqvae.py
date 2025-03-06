import torchimport torch.nn.functional as Ffrom constants import *from utils import *fix_esm_path()import esmimport torchfrom rosetta_former.energy_vqvae import *from dataset import *# HyperparametersENERGY_INPUT_DIM = 20ENERGY_D_OUT = 512ENERGY_D_MODEL = 1024ENERGY_N_CODEBOOK = 16384# 8192ENCODER_DEPTH=4DECODER_DEPTH=16commitment_cost = .5batch_size = 32num_epochs = 10dataset_name = "gfp_train_1"def train(model_name, model, optimizer, train_data_loader):        dataset = RawTokensDataset(dataset_name, mode="energy")    # Training loop    for epoch in range(num_epochs):        ctr = 0                save_torch_model(model_name, model, optimizer)                for batch in train_data_loader:                        ctr += 1            optimizer.zero_grad()            # Forward pass            x_recon, vq_loss = model(batch)            # Reconstruction loss            recon_loss = F.mse_loss(x_recon, batch)            # Total loss            loss = recon_loss + vq_loss                        print("\t(E:%d) %d loss: %.3f [recon %.3f, vqvae %.3f]" % (epoch + 1, ctr + 1, loss.item(), recon_loss.item(), vq_loss.item()))            loss.backward()            optimizer.step()        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")    save_torch_model(model_name, model, optimizer)    print("Training completed!")dataset = RawTokensDataset(dataset_name, mode="energy")train_data_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)# Initialize model, optimizer, and loss functionmodel = EnergyVQVAE(ENERGY_INPUT_DIM,                     ENERGY_D_MODEL,                     ENERGY_D_OUT,                     ENERGY_N_CODEBOOK,                     commitment_cost,                    ENCODER_DEPTH,                    DECODER_DEPTH)optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)# train("energy_vqvae_v0_deeper", model, optimizer, train_data_loader)# model = load_torch_model("energy_vqvae_v1", model)# x = dataset[5]# a,b=model(x.unsqueeze(dim=0))# rec = a.squeeze()# [scipy.stats.pearsonr(x[i,:].detach().numpy(), rec[i,:].detach().numpy())[0] for i in range(225)]