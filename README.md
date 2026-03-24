# Uncertainty Quantification for VERaiPHY

This repository contains reproducible toy examples, figures, and Jupyter notebooks accompanying the VERaiPHY article on uncertainty quantification in machine learning.

The material is intended to illustrate core uncertainty quantification concepts in simple and controlled settings, with a particular focus on regression, classification, and a Bernstein--von Mises example.

## Contents

The repository is organised into a small number of self-contained example folders:

- `regression/`  
  Toy regression examples and notebooks illustrating predictive uncertainties, calibration, coverage, and related diagnostics.

- `classification/`  
  Toy classification examples and notebooks illustrating uncertainty-aware classification and associated validation tools.

- `bvm_example/`
  Material related to the Bernstein--von Mises example used in Section 3, currently provided as the notebook `chapter3_bvm.ipynb`.

## Purpose

This repository is not intended as a Python library.  
Instead, it serves as a compact and reproducible collection of examples supporting the discussion in the associated article.

The emphasis is on clarity and transparency rather than software packaging: most content is provided in the form of one or more Jupyter notebooks inside each example directory.

## Setup

The examples require Python 3.12 or newer.
After creating and activating a suitable Python environment, install the dependencies listed in `pyproject.toml`.
For example, with `pip`:

```bash
pip install -e .