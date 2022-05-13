# Prototypical multiple instance learning for predicting lymph node metastasis of breast cancer from whole-slide pathological images

<img src="https://github.com/Zero-We/PMIL/blob/main/docs/pmil-overview.png">


## Introduction
Computerized identification of lymph node metastasis (LNM) from whole-slide pathological images (WSIs) can largely benefit the therapy decision and prognosis of breast cancer. Besides the general challenges of computational pathology, including extra high resolution, very expensive fine-grained annotation and significant inter-tumoral heterogeneity, one particular difficulty with this task lies in identifying metastasized tumors with tiny foci (called micro-metastasis). In this study, we introduce a weakly supervised method, called Prototypical Multiple Instance Learning (PMIL), to learn to predict lymph node metastasis of breast cancer from whole slide pathological images with only slide-level class labels. Firstly, PMIL discovers a collection of so-called prototypes from the training data by unsupervised clustering. Secondly, the prototypes are matched against the constitutive patches in the WSI, and the resultant similarity scores are aggregated into a soft-assignment histogram describing the statistical distribution of the prototypes in the WSI, which is taken as the slide features. Finally, WSI classification is accomplished by using the slide features.


## Dataset
* **Camelyon16**  
Camelyon16 is a public challenge dataset of sentinel lymph
node biopsy of early-stage breast cancer, which includes 270 H&E-stained WSIs for training and 129 for testing (48 LNM-positive and 81 LNM-negative), collected from two medical centers.   
Download from [here](https://camelyon17.grand-challenge.org/Data/).

* **Zbraln**  
The Zhujiang Breast Cancer Lymph Node (Zbraln) was created by ourselves. Specifically, we collected 635 H&E-stained glass slides of dissected ALNs.  
We only provide a few whole slide images data here due to the privacy policy. [[Google Drive]](https://drive.google.com/drive/folders/1BS5ZVOJKPbGM7zRSKIlvEyC2jaQ_tK9E?usp=sharing, https://drive.google.com/drive/folders/1bQBFaTeGDAzhYg6w6GJfvCnzE-atvUyT?usp=sharing) | [[Baidu Cloud]](https://pan.baidu.com/s/1r-D7o1J9pmebOSi9s0Bm5w) (fzts)

## Evaluation
    python pmil.py --train_lib 'lib/train.ckpt' --val_lib '' --test_lib 'lib/test.ckpt' --train_feature_dir 'feat' --test_feature_dir 'feat' --output 'result' --global_cluster 'cluster/prototypes_features_40x256.npy' --mil_model 'model/checkpoint_best_40x256.pth' --pmil 'model/pmil_model.pth' --suffix '.csv' --load_model --is_test
