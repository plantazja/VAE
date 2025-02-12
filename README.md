# Statistical Data Analysis 2 - Project 2

Applying different Variational Autoencoders (VAEs) to biological data by customizing a VAE using probabilistic modelling.

## Usage
Script that implements Variational Autoencoder, trains on scRNA-seq data and saves the models.

`python3 train.py -i data_dir -o output_dir [--use_gpu]`

where
* `-i`  Input directory with data .h5ad data.
* `-o`  Output directory where the trained model and learning curve plots will be saved.
* `--use_gpu` Use GPU for training (if available)

Script that explores scRNA-seq from .h5ad files, loads the trained model and visualizes latent space using UMAP.

`python3 eval.py -i data_dir -o output_dir -m model_file [--use_gpu]`
where
* `-i`  Input directory with data .h5ad data.
* `-o`  Output directory where report and figures will be saved.
* `-m` Pretrained model in .ptx format.
* `--use_gpu` Use GPU (if available)