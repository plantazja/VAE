import anndata
import os
import matplotlib.pyplot as plt
import scanpy as sc
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script with implementation of VAE and training on AnnData")
    parser.add_argument("-i", "--input_data", type=str, required=True, help="Input directory with data .h5ad data")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Path to the directory where the trained model and learning curve plots will be saved.")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU for training (if available)")
    return parser.parse_args()

class Encoder(nn.Module):
    '''
    Takes gene expression data and passes it through a series
     of fully connected linear layers with ReLU activation.
    '''
    def __init__(self, input_size, hidden_sizes, latent_size):
        super().__init__()
        self.layers = nn.ModuleList()

        prev_size = input_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.fc_mu = nn.Linear(prev_size, latent_size)  # output layers for mean
        self.fc_logvar = nn.Linear(prev_size, latent_size)  # output layers for log variance

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar  # mean and log variance of the latent distribution

class Decoder(nn.Module):
    '''
    Takes a latent vector z and passes it through
    a series of fully connected layers with ReLU activation.
    '''
    def __init__(self, latent_size, hidden_sizes, output_size):
        super().__init__()
        self.layers = nn.ModuleList()

        prev_size = latent_size
        for hidden_size in hidden_sizes:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size

        self.fc_out = nn.Linear(prev_size, output_size)  # output layer

    def forward(self, z):
        for layer in self.layers:
            z = F.relu(layer(z))
        x_recon = torch.sigmoid(self.fc_out(z))  # sigmoid ensures output is in [0, 1]
        return x_recon  # reconstructed matrix

class VAE(nn.Module):
    def __init__(self, input_size, hidden_sizes, latent_size):
        super().__init__()
        self.encoder = Encoder(input_size, hidden_sizes, latent_size)
        self.decoder = Decoder(latent_size, hidden_sizes, input_size)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std  # sampled latent vector

    def sample_latent(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return z  # sample latent vector from the approximate posterior distribution

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar  # reconstructed matrix, mean and log variance of the latent distribution

    def get_latent_representation(self, x): # for UMAP
        mu, _ = self.encoder(x)
        return mu

def kl_divergence(mu, logvar):
    """
    Compute the KL divergence between q(z|x) and p(z) """
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean()  # average over the batch

def elbo(x_recon, x, mu, logvar):
    """
        x_recon: reconstructed output from the Decoder;
        x: input data;
        mu: mean of latent distribution;
        logvar: Log variance of the latent distribution;
    """
    recon_loss = F.mse_loss(x_recon, x, reduction='sum')  # Sum over all elements
    kl = kl_divergence(mu, logvar)
    elbo_value = recon_loss + kl
    return elbo_value, recon_loss, kl

def train_vae(model, train_loader, test_loader,epochs=100, lr=1e-3,device="cpu"):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_elbo, test_elbo = [], []
    train_kl, test_kl = [], []
    train_recon, test_recon = [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x in train_loader:
            x = x[0].to(device)
            optimizer.zero_grad()
            x_recon, mu, logvar = model(x)
            loss, recon_loss, kl = elbo(x_recon, x, mu, logvar)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # append losses and ensure they are on the CPU
        train_elbo.append(train_loss / len(train_loader))
        train_kl.append(kl.item() if torch.is_tensor(kl) else kl)
        train_recon.append(recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss)


        model.eval()
        test_loss = 0
        with torch.no_grad():
            for x in test_loader:
                x = x[0].to(device)
                x_recon, mu, logvar = model(x)
                loss, recon_loss, kl = elbo(x_recon, x, mu, logvar)
                test_loss += loss.item()

        # append losses and ensure they are on the CPU
        test_elbo.append(test_loss / len(test_loader))
        test_kl.append(kl.item() if torch.is_tensor(kl) else kl)
        test_recon.append(recon_loss.item() if torch.is_tensor(recon_loss) else recon_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {train_elbo[-1]}, Test Loss: {test_elbo[-1]}")

    return train_elbo, test_elbo, train_kl, test_kl, train_recon, test_recon

def process_data(adata, layer='counts', target_sum=1e4):
    ''' Processing steps (normalization and log transformation) '''
    adata.layers['counts_processed'] = adata.layers[layer].copy()
    sc.pp.normalize_total(adata, target_sum=target_sum, layer='counts_processed')
    sc.pp.log1p(adata, layer='counts_processed')

def load_data(dataset, device="cpu"):
    train_data, test_data = dataset

    train_tensor = torch.tensor(train_data.toarray(), dtype=torch.float32).to(device)
    test_tensor = torch.tensor(test_data.toarray(), dtype=torch.float32).to(device)

    # create TensorDataset
    train_dataset = TensorDataset(train_tensor)
    test_dataset = TensorDataset(test_tensor)

    # create DataLoader
    batch_size = 128
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader

def plot_losses(name, output_dir, train_losses_dict, test_losses_dict, loss_type):
    latent_sizes = [16, 32, 64]

    plt.figure(figsize=(14, 6))
    colormap = plt.get_cmap('cool')
    norm = plt.Normalize(vmin=min(latent_sizes), vmax=max(latent_sizes))

    for latent_size in latent_sizes:
        color = colormap(norm(latent_size))

        train_losses = train_losses_dict[latent_size]
        test_losses = test_losses_dict[latent_size]

        if torch.is_tensor(train_losses):
            train_losses = train_losses.cpu().numpy()
        if torch.is_tensor(test_losses):
            test_losses = test_losses.cpu().numpy()

        plt.plot(train_losses, label=f"Train Loss (Latent Size={latent_size})", color=color, linestyle='-')
        plt.plot(test_losses, label=f"Test Loss (Latent Size={latent_size})", color=color, linestyle='--')

    plt.xlabel("Epochs")
    plt.ylabel(f"{loss_type}")
    plt.yscale('log')
    plt.legend()
    plt.title(f"Learning Curves for {name}")
    plot_path = os.path.join(output_dir, f"learning_curve_{name}_{loss_type}.png")
    plt.savefig(plot_path)
    plt.close()

def table_losses(output_dir, dataset_name, latent_sizes,
                 elbo_train_dict, elbo_test_dict,
                 kl_train_dict, kl_test_dict,
                 recon_train_dict, recon_test_dict):
    results = []

    for latent_size in latent_sizes:
        train_elbo = elbo_train_dict[latent_size][-1]
        test_elbo = elbo_test_dict[latent_size][-1]
        train_kl = kl_train_dict[latent_size][-1]
        test_kl = kl_test_dict[latent_size][-1]
        train_recon = recon_train_dict[latent_size][-1]
        test_recon = recon_test_dict[latent_size][-1]

        # Append the results to the list
        results.append({
            "Dataset": dataset_name,
            "Latent Size": latent_size,
            "Train ELBO": train_elbo,
            "Test ELBO": test_elbo,
            "Train KL": train_kl,
            "Test KL": test_kl,
            "Train Recon": train_recon,
            "Test Recon": test_recon
        })

    results_df = pd.DataFrame(results)

    table_path = os.path.join(output_dir, f"loss_summary_{dataset_name}.tsv")
    results_df.to_csv(table_path, sep="\t", index=False)

    print(f"Loss summary table saved to {table_path}")

def main():
    args = parse_arguments()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load datasets
    for file in os.listdir(args.input_data):
        if file.endswith("test.h5ad"):
            adata_test = anndata.read_h5ad(os.path.join(args.input_data, file))
        elif file.endswith("train.h5ad"):
            adata_train = anndata.read_h5ad(os.path.join(args.input_data, file))

    # Determine device
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Process and save in layer ['counts_processed']
    process_data(adata_train)
    process_data(adata_test)

    # Define datasets
    datasets = {
        "Raw_matrix": [adata_train.layers['counts'], adata_test.layers['counts']],
        "Custom_matrix": [adata_train.layers['counts_processed'], adata_test.layers['counts_processed']]
    }

    datasets_loader = {k: load_data(v, device) for k, v in datasets.items()}
    latent_sizes = [16, 32, 64]

    # Model training for each dataset
    for dataset_name, loader in datasets_loader.items():
        """ Training for each dataset and latent size, results (training curves
         and summary table) are saved in directory output_dir/dataset_name """

        print(f"Training on dataset: {dataset_name}")
        train_loader, test_loader = loader
        dataset_output = os.path.join(args.output_dir, dataset_name)
        os.makedirs(dataset_output, exist_ok=True)

        elbo_train_dict, elbo_test_dict = {}, {}
        kl_train_dict, kl_test_dict = {}, {}
        recon_train_dict, recon_test_dict = {}, {}

        # Train VAE for different latent sizes
        for latent_size in latent_sizes:
            model = VAE(input_size=adata_train.n_vars, hidden_sizes=[256, 128], latent_size=latent_size).to(device)
            train_elbo, test_elbo, train_kl, test_kl, train_recon, test_recon = train_vae(model, train_loader,test_loader,
                                                                                          epochs=100, lr=1e-4, device=device)
            torch.save(model.state_dict(), os.path.join(dataset_output, f"vae_model_{latent_size}.pth"))
            elbo_train_dict[latent_size], elbo_test_dict[latent_size] =  train_elbo, test_elbo
            kl_train_dict[latent_size], kl_test_dict[latent_size] = train_kl, test_kl
            recon_train_dict[latent_size], recon_test_dict[latent_size] = train_recon, test_recon

            print(f"Model saved to {dataset_output}/vae_model_{latent_size}.pth")

        plot_losses(dataset_name, dataset_output, elbo_train_dict, elbo_test_dict, loss_type='ELBO_Loss')
        plot_losses(dataset_name, dataset_output, kl_train_dict, kl_test_dict, loss_type='Kullback_Liebler_Divergence')
        plot_losses(dataset_name, dataset_output, recon_train_dict, recon_test_dict, loss_type='Reconstruction_Loss')

        # not a perfect function
        table_losses(dataset_output, dataset_name, latent_sizes,
                     elbo_test_dict, elbo_train_dict,
                     kl_train_dict, kl_test_dict,
                     recon_train_dict, recon_test_dict)


if __name__ == "__main__":
    main()