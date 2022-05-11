# Prototypical multiple instance learning for predicting lymph node metastasis of breast cancer from whole-slide pathological images

<img src="https://github.com/Zero-We/PMIL/blob/main/docs/pmil-overview.png" width="800px">


## Introduction
Computerized identification of lymph node metastasis (LNM) from whole-slide pathological images (WSIs) can largely benefit the therapy decision and prognosis of breast cancer. Besides the general challenges of computational pathology, including extra high resolution, very expensive fine-grained annotation and significant inter-tumoral heterogeneity, one particular difficulty with this task lies in identifying metastasized tumors with tiny foci (called micro-metastasis). In this study, we introduce a weakly supervised method, called Prototypical Multiple Instance Learning (PMIL), to learn to predict lymph node metastasis of breast cancer from whole slide pathological images with only slide-level class labels. Firstly, PMIL discovers a collection of so-called prototypes from the training data by unsupervised clustering. Secondly, the prototypes are matched against the constitutive patches in the WSI, and the resultant similarity scores are aggregated into a soft-assignment histogram describing the statistical distribution of the prototypes in the WSI, which is taken as the slide features. Finally, WSI classification is accomplished by using the slide features.

## Evaluation
