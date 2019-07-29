# Trimming the ℓ₁ Regularizer: Statistical Analysis, Optimization, and Applications to Deep Learning
+ Jihun Yun (KAIST), Peng Zheng (University of Washington), Eunho Yang (KAIST, AITRICS), Aurélie C. Lozano (IBM T.J.
Watson Research Center), and Aleksandr Aravkin (University of Washington)

This repo contains implementations for our **ICML 2019** paper "Trimming the ℓ₁ Regularizer: Statistical Analysis, Optimization, and Applications to Deep Learning".

# Abstract

We study high-dimensional estimators with the trimmed ℓ1 penalty, which leaves the h largest parameter entries penalty-free. While optimization techniques for this nonconvex penalty have been studied, the statistical properties have not yet been analyzed. We present the first statistical analyses for M-estimation, and characterize support recovery, ℓ∞ and ℓ2 error of the trimmed ℓ1 estimates as a function of the trimming parameter h. Our results show different regimes based on how h compares to the true support size. Our second contribution is a new algorithm for the trimmed regularization problem, which has the same theoretical convergence rate as difference of convex (DC) algorithms, but in practice is faster and finds lower objective values. Empirical evaluation of ℓ1 trimming for sparse linear regression and graphical model estimation indicate that trimmed ℓ1 can outperform vanilla ℓ1 and non-convex alternatives. Our last contribution is to show that the trimmed penalty is beneficial beyond M-estimation, and yields promising results for two deep learning tasks: input structures recovery and network sparsification.

# Citation

If you think this repo is helpful, please cite as
```
@inproceedings{yun2019trimming,
  title={Trimming the $$\backslash$ell\_1 $ Regularizer: Statistical Analysis, Optimization, and Applications to Deep Learning},
  author={Yun, Jihun and Zheng, Peng and Yang, Eunho and Lozano, Aurelie and Aravkin, Aleksandr},
  booktitle={International Conference on Machine Learning},
  pages={7242--7251},
  year={2019}
}
```
