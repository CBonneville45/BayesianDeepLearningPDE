# Bayesian Deep Learning for PDE Discovery

Scientific machine learning has been successfully applied to inverse problems and PDE discovery in computational physics. One caveat concerning current methods is the need for large amounts of (``clean") data, in order to characterize the full system response and discover underlying physical models. Bayesian methods may be particularly promising for overcoming these challenges, as they are naturally less sensitive to the negative effects of sparse and noisy data. In this paper, we propose to use Bayesian neural networks (BNN) in order to: 1) Recover the full system states from measurement data (e.g. temperature, velocity field, etc.). We use Hamiltonian Monte-Carlo to sample the posterior distribution of a deep and dense BNN, and show that it is possible to accurately capture physics of varying complexity, without overfitting. 2) Recover the parameters instantiating the underlying partial differential equation (PDE) governing the physical system. Using the trained BNN, as a surrogate of the system response, we generate datasets of derivatives that are potentially comprising the latent PDE governing the observed system and then perform a sequential threshold Bayesian linear regression (STBLR), between the successive derivatives in space and time, to recover the original PDE parameters. We take advantage of the confidence intervals within the BNN outputs, and introduce the spatial derivatives cumulative variance into the STBLR likelihood, to mitigate the influence of highly uncertain derivative data points; thus allowing for more accurate parameter discovery. We demonstrate our approach on a handful of example, in applied physics and non-linear dynamics.

The full paper can be found at: https://arxiv.org/abs/2108.04085

This repository contains codes and data for reproducing the paper results.

## Dependencies

The code requires:
* **PyTorch 1.7.1** 
* **Hamiltorch 0.4.0.dev1** - https://github.com/AdamCobb/hamiltorch
* **NumPy** 
* **Scikit-Learn**

## Code Summary

The code contains the following files:
* **TrainingBNN/TrainingBNN.py** and **TrainingDNN/TrainingDNN.py** which are wrapping class containing neural networks training framework
* **GenerateDerivatives/GenerateDerivatives.py** which is are class that performs fast auto-differentiation using Torch. It takes trained (Bayesian) neural networks and space-time coordinates and computes dataset of successive derivatives
* **BurgersEquation**, **KdVEquation** and **HeatEquation**, which contains the three examples introduced in the paper. Each folder contains the original PDE solution data, the generated sensor measurement data, the codes for calling **TrainingBNN.py**, **TrainingDNN.py** and **GenerateDerivatives.py**, the derivative data, and the codes for sequential threshold regressions. The Bayesian neural networks samples obtained through HMC sampling can be downloaded at https://drive.google.com/drive/folders/19FnCYVlAMj3X9FnwnJlD0J_UF8MGi-Lt?usp=sharing

## References

```
@misc{bonneville2021bayesian,
      title={Bayesian Deep Learning for Partial Differential Equation Parameter Discovery with Sparse and Noisy Data}, 
      author={Christophe Bonneville and Christopher J. Earls},
      year={2021},
      eprint={2108.04085},
      archivePrefix={arXiv},
      primaryClass={math.NA}
}
```

