import anndata
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import scanpy as sc
import torch
from train import VAE

def parse_arguments():
    parser = argparse.ArgumentParser(description="Script that loads the trained model and reproduces all the results and figures included in the report")
    parser.add_argument("-i", "--input_data", type=str, required=True, help="Input directory with data .h5ad data")
    parser.add_argument("-o", "--output_dir", type=str, required=True, help="Output directory where report and figures will be saved")
    parser.add_argument("-m", "--model_path", type=str, required=False, help="Path to the Model to use")
    parser.add_argument("--use_gpu", action="store_true", help="Use GPU (if available)")
    return parser.parse_args()

def data_exploration(test, train, output_path):
    with open(os.path.join(output_path, "data_summary.txt"), "w") as f:
        f.write(f"DATA EXPLORATION\n\n")
        f.write(f"{train}\n\n")
        f.write(f"Training dataset: {train.n_obs} observations (cells), {train.n_vars} variables (genes).\n")
        f.write(f"Test dataset: {test.n_obs} observations (cells), {test.n_vars} variables (genes).\n\n")

        # Number of patients
        num_patients_test = test.obs['DonorID'].nunique()
        f.write(f"Number of patients: {num_patients_test}\n")

        # Information available about patients
        patient_col = [col for col in test.obs.columns if col in ['DonorID', 'DonorAge', 'DonorBMI', 'DonorBloodType',
                                                              'DonorRace', 'Ethnicity', 'DonorGender', 'QCMeds', 'DonorSmoker']]
        f.write("Information available about patients:\n")
        f.write(f"{test.obs[patient_col].head()}\n\n")

        # Information available about cells
        cell_col = [col for col in test.obs.columns if col in ['cell_type', 'batch', 'GEX_n_genes_by_counts', 'GEX_pct_counts_mt',
                                                               'GEX_phase', 'ADT_n_antibodies_by_counts', 'ADT_total_counts']]
        f.write("\nInformation available about cells:\n")
        f.write(f"{test.obs[cell_col].head()}\n\n")

        # Number of cell types
        num_cell_types = test.obs['cell_type'].nunique()
        f.write(f"\nNumber of cell types: {num_cell_types}\n")

        # Number of patients
        num_lab = test.obs['Site'].nunique()
        f.write(f"Number of laboratories where samples were prepared: {num_lab}\n")

        # Number of batches
        num_batches = test.obs['batch'].nunique()
        f.write(f"Number of batches: {num_batches}\n\n")

def plot_cell_distribution(adata, output_path):
    cell_type_counts = adata.obs.groupby(['DonorID', 'cell_type']).size().unstack(fill_value=0)

    cell_type_order = cell_type_counts.sum().sort_values(ascending=False).index
    cell_type_counts = cell_type_counts[cell_type_order]

    ax = cell_type_counts.plot(kind='barh', stacked=True, figsize=(10, 8))
    ax.set_title('Cell types distribution', fontsize=16)
    ax.set_xlabel("Number of cells", fontsize=12)
    ax.set_ylabel("Patient ID", fontsize=12)
    ax.legend(title="Cell types", bbox_to_anchor=(1, 1), loc='upper left', fontsize=7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, f"task1_4.png"))

def plot_expression_distribution(adata, output_path):
    raw_counts = adata.layers['counts'].toarray().flatten()
    processed_counts = adata.X.toarray().flatten()

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(raw_counts, bins=100, color='skyblue')
    plt.title("Raw counts distribution")
    plt.xlabel("Count values")
    plt.ylabel("Frequency")
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    plt.hist(processed_counts, bins=100, color='pink')
    plt.title("Processed counts distribution")
    plt.xlabel("Count values")
    plt.ylabel("Frequency")
    plt.yscale('log')

    plt.savefig(os.path.join(output_path, f"task1_5_1.png"))

    # Report ranges and zero counts
    with open(os.path.join(output_path, "data_summary.txt"), "a") as f:
        f.write(f"Number of zeros in count matrix: {np.sum(raw_counts == 0)}\n")
        f.write("====================================================================================================")

    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    processed_counts = adata.X.toarray().flatten()

    plt.figure(figsize=(14, 6))

    plt.subplot(1, 2, 1)
    plt.hist(raw_counts, bins=100,color='skyblue')
    plt.title("Raw counts distribution")
    plt.xlabel("Count values")
    plt.ylabel("Frequency")
    plt.yscale('log')

    plt.subplot(1, 2, 2)
    plt.hist(processed_counts, bins=100, color='pink')
    plt.title("Processed counts distribution")
    plt.xlabel("Count values")
    plt.ylabel("Frequency")
    plt.yscale('log')
    plt.savefig(os.path.join(output_path, f"task1_5_2.png"))


def plot_umap(model, output_path, adata_test, color_by, device):
    model.eval()
    with torch.no_grad():
        x = torch.tensor(adata_test.X.toarray(), dtype=torch.float32).to(device)
        Z_hat = model.get_latent_representation(x).cpu().numpy()

    # Add latent representation to adata_test
    for i, z in enumerate(Z_hat.T):
        adata_test.obs[f"Z_{i}"] = z

    adata_test.obsm["X_scVI"] = Z_hat

    plt.figure(figsize=(20, 10))
    sc.pp.neighbors(adata_test, use_rep="X_scVI", n_neighbors=20)
    sc.tl.umap(adata_test, min_dist=0.3, random_state=None)
    sc.tl.leiden(adata_test, key_added="leiden", resolution=0.8)

    plt.figure(figsize=(12, 8))
    sc.pl.umap(adata_test, color=[color_by], show=False)
    plt.tight_layout()

    plt.savefig(
        os.path.join(output_path, f"umap_{color_by}.png"),
        dpi=300,
        bbox_inches='tight'
    )
    plt.close()


def main():
    args = parse_arguments()

    os.makedirs(args.output_dir, exist_ok=True)

    for file in os.listdir(args.input_data):
        if file.endswith("test.h5ad"):
            adata_test = anndata.read_h5ad(os.path.join(args.input_data, file))
        elif file.endswith("train.h5ad"):
            adata_train = anndata.read_h5ad(os.path.join(args.input_data, file))

    adata_merged = anndata.concat([adata_train, adata_test], axis=0, join='inner')

    # Data Exploration
    data_exploration(adata_test, adata_train, args.output_dir)
    plot_cell_distribution(adata_merged, args.output_dir)
    plot_expression_distribution(adata_merged, args.output_dir)

    # VAE evaluation
    device = torch.device("cuda" if args.use_gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    sc.pp.normalize_total(adata_test)
    sc.pp.log1p(adata_test)

    model = VAE(adata_test.n_vars, [256, 128], 32).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    color_by_list = ['cell_type', 'DonorNumber', 'batch', 'Site']

    for inf_type in color_by_list:
        plot_umap(model, args.output_dir, adata_test, color_by=inf_type, device=device)

if __name__ == "__main__":
    main()