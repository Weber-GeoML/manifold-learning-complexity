# Hardness of Learning Neural Networks under the Manifold Hypothesis

This repository contains all the code and scripts that reproduce the results in our paper.

The experiments consist of two sections:

1. *Empirical Verification of Main Findings:* Learnability of neural networks in easy (sampleable) and hard (bounded curvature) regimes

2. *Empirical Study of Geometry of Data Manifolds:* Assessing the intrinsic dimension to characterize the middle regime in which real data likely falls under.

## Setup

```
conda create -n manifold-ml python=3.10
conda activate manifold-ml
pip install torch=2.2.0 torchvision=0.17.0 torchaudio=2.2.0
pip install diffusers=0.27.2
pip install datasets=2.18.0
pip install accelerate=0.28.0
pip install scikit-learn=1.4.1
pip install matplotlib=3.8.3
```

## Reproduction

To reproduce Experiment 1, please run the following two Jupyter notebooks:

- Easy Regime: `manifold_nn_experiments_isoperimetry.ipynb` 

- Hard Regime: `manifold_nn_experiments_parity.ipynb`

To reproduce Experiment 2, please run the following shell script:

- `intrinsic_dimension_experiments.sh`

Note that to run Experiment 2 on KMNIST, you must first [download](https://github.com/rois-codh/kmnist) the KMNIST dataset into `datasets/kmnist/[xx].png`.

All experiments can be accelerated with a GPU. We used an NVIDIA L4 24GB GPU, which was more than enough for our experiments.

## Other Code Files

`train_diffusion_model.py` trains a diffusion model on an image dataset, and `train_diffusion_model_synthetic.py` trains a diffusion model on a hypersphere dataset.

`generate_from_diffusion.py` contains the procedures for generating samples from and finding the intrinsic dimension around an image with a trained diffusion model, and `generate_from_diffusion_synthetic.py` contains the procedures for generating samples from and finding the intrinsic dimension around a point with a trained diffusion model.

`data.py` contains the code for generating a hypersphere with arbtirary intrinsic and ambient dimension.

`models.py` contains the code for a fully connected neural network used in the diffusion model for the hypersphere.

## Licenses and Citations

Experiment 2 (mostly) follows the algorithm in [Stanczuk et al. 2022](https://arxiv.org/abs/2212.12611).

```
Stanczuk, Jan, et al. "Your diffusion model secretly knows the dimension of the data manifold." arXiv preprint arXiv:2212.12611 (2022).
```

Experiment 2 also uses three open-source MNIST-esque datasets:

- [MNIST](yann.lecun.com/exdb/mnist/): `LeCun, Yann, Corinna Cortes, and CJ Burges. "MNIST Handwritten Digit Database." ATT Labs, vol. 2, 2010, http://yann.lecun.com/exdb/mnist.`
    - Creative Commons Attribution-Share Alike 3.0
- [FMNIST](https://github.com/zalandoresearch/fashion-mnist): `Xiao, Han, Kashif Rasul, and Roland Vollgraf. "Fashion-MNIST: A Novel Image Dataset for Benchmarking Machine Learning Algorithms." arXiv preprint arXiv:1708.07747 (2017).`
    - MIT License
- [KMNIST](https://github.com/rois-codh/kmnist): `Clanuwat, Tarin, et al. "Deep Learning for Classical Japanese Literature." arXiv preprint arXiv:1812.01718 (2018).`
    - Creative Commons Attribution Share Alike 4.0
