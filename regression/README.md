# Regression examples

This folder contains the toy regression benchmarks accompanying the VERaiPHY uncertainty quantification study.

The examples are designed to illustrate and compare different approaches to predictive uncertainty quantification in a simple supervised regression setting. In particular, the notebooks cover Bayesian neural networks, Gaussian processes, repulsive ensembles, and conformal prediction, together with standard diagnostics such as calibration and pull distributions.

## Contents

### Model notebooks

- `bnn_toy.ipynb`  
  Toy regression example using a Bayesian neural network.

- `gp_toy.ipynb`  
  Toy regression example using a Gaussian process.

- `repulsive_toy.ipynb`  
  Toy regression example using a repulsive ensemble.

- `conformal_toy.ipynb`  
  Toy regression example using conformal prediction.

### Plotting notebook

- `plots.ipynb`  
  Produces the combined calibration and pull plots for the different regression methods.

## Results

For convenience, this folder also contains precomputed results for the four model classes in

- `results/*_results.npy`

These files are used by `plots.ipynb` to generate the combined comparison plots without requiring all model notebooks to be rerun.

## Usage

The individual model notebooks can be run independently to reproduce the corresponding regression examples and saved outputs.

To reproduce the combined comparison plots, open and run

- `plots.ipynb`

which reads the precomputed files from the `results/` subdirectory.

## Notes

The notebooks are intended as illustrative toy examples accompanying the article, rather than as a standalone software package. The emphasis is therefore on clarity and reproducibility of the examples.