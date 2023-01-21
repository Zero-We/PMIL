# Prototypical multiple instance learning for predicting lymph node metastasis of breast cancer from whole-slide pathological images

<img src="https://github.com/Zero-We/PMIL/blob/main/docs/pmil-overview.png">


## Introduction
Computerized identification of lymph node metastasis (LNM) from whole-slide pathological images (WSIs) can largely benefit the therapy decision and prognosis of breast cancer. Besides the general challenges of computational pathology, including extra high resolution, very expensive fine-grained annotation and significant inter-tumoral heterogeneity, one particular difficulty with this task lies in identifying metastasized tumors with tiny foci (called micro-metastasis). In this study, we introduce a weakly supervised method, called Prototypical Multiple Instance Learning (PMIL), to learn to predict lymph node metastasis of breast cancer from whole slide pathological images with only slide-level class labels. Firstly, PMIL discovers a collection of so-called prototypes from the training data by unsupervised clustering. Secondly, the prototypes are matched against the constitutive patches in the WSI, and the resultant similarity scores are aggregated into a soft-assignment histogram describing the statistical distribution of the prototypes in the WSI, which is taken as the slide features. Finally, WSI classification is accomplished by using the slide features.
<br/>

## Model
The trained model weights and precomputed patch feature vectors are provided here （[[Google Drive]](https://drive.google.com/drive/folders/1kfib8H-4jhNzwj-_LDmUGVtjCv3Lg6zT?usp=sharing) | [[Baidu Cloud]](https://pan.baidu.com/s/1OQJM8Tp7y1RlRIPUKdjqIA) (fzts)）. You can download these files and drag `pmil_model.pth` and `pmil_model_simclr.pth` to  the `model` directory, drag `mil-feat` and `simclr-feat` to the `feat` directory.  
<br/>

## Dataset
* **Camelyon16**  
Camelyon16 is a public challenge dataset of sentinel lymph
node biopsy of early-stage breast cancer, which includes 270 H&E-stained WSIs for training and 129 for testing (48 LNM-positive and 81 LNM-negative), collected from two medical centers.   
Download from [here](https://camelyon17.grand-challenge.org/Data/).

* **Zbraln**  
The Zhujiang Breast Cancer Lymph Node (Zbraln) was created by ourselves. Specifically, we collected 635 H&E-stained glass slides of dissected ALNs.  
We only provide a few whole slide images data here due to the privacy policy. [[Google Drive]](https://drive.google.com/drive/folders/1kfib8H-4jhNzwj-_LDmUGVtjCv3Lg6zT?usp=sharing) | [[Baidu Cloud]](https://pan.baidu.com/s/1OQJM8Tp7y1RlRIPUKdjqIA) (fzts)  
<br/>

## Training  
The patch-level feature encoder will be initialized by training the standard instance-space MIL with max-pooling. Part of our code refer to: (Campanella et al., 2019), you can refer to [here](https://github.com/MSKCC-Computational-Pathology/MIL-nature-medicine-2019). And the input data should be stored in dictionary with `torch.save()` in `.ckpt` file format including following keys:  
* `'slides'`: a list of paths to WSIs.  
* `'grid'`: a list of patch coordinates tuple (x,y). Size of the list equal to number of slides, and the size of each sublist is equal to the numbers of patches in each slide.  
* `'target'`: a list of slide-level target.  
* `'mult'`: scale factor for achieving resolutions different than the ones saved in WSI pyramid file.
* `'level'`: WSI level to tile the patches.  

You can run following command to train the standard MAX-MIL model and extract the feature vectors of each patch simultaneously:  
~~~
python max-mil.py --save_model --save_index --save_feat
~~~  
<br/>
  

Affinity propagation clustering algorithm is used to capture the typical pathological patterns, which we call prototypes. To obtain the prototypes on Camelyon16 dataset, you can run following command:  
~~~
python cluster.py
~~~  
<br/>
  

Train the PMIL framework that encodes WSI by its compositions in terms of the frequencies of occurence of prototypes found inside. Here, we use patch features match against prototypes to get soft-assignment histogram, andd histograms of each patch in WSI will be aggregated by selective pooling module:  
~~~
python pmil.py --save_model
~~~  
<br/>
  
  

## Inference  
You can evaluate the performance of PMIL at 40x magnification on Camelyon16 dataset by following command: 
~~~
python pmil.py --load_model --is_test
~~~  
<br/>

## Visualization
We illustare the prototype discovery on Camelyon16 dataset here. The above row of images show the discovered prototypes, and the colors of bounding boxes are matched with the colors of each cluster in the below row. The below shows intra-slide patch clustering results on two WSIs, the left is LNM-positive and the right is LNM-negative.  
<div align=center><img src="https://github.com/Zero-We/PMIL/blob/main/docs/prototype-discovery.png" width="800px"></div>
<br/>

Interpretablity is important to deep learning based algorithms for medical applications, fow which MIL methods often utilize a so-called heatmap to visualize the contribution of each location in a WSI to the classification decision. And we also illustrate the attention maps obtained by PMIL in the `vis` directory. We can observe that, the attention map can completely highlight the tumor regions, which are consistent with the ground truth annotations.  
<div align=center><img src="https://github.com/Zero-We/PMIL/blob/main/docs/attention-map.png" width="800px"></div>
<br/>

## License  
This code is made available under the GPLv3 License and is available for non-commercial academic purposes.
<br/>

## Citation  
If you find our work useful in your research or if you use parts of this code please consider citing our paper.  
~~~
@article{yu2023prototypical,
  title={Prototypical multiple instance learning for predicting lymph node metastasis of breast cancer from whole-slide pathological images},
  author={Yu, Jin-Gang and Wu, Zihao and Ming, Yu and Deng, Shule and Li, Yuanqing and Ou, Caifeng and He, Chunjiang and Wang, Baiye and Zhang, Pusheng and Wang, Yu},
  journal={Medical Image Analysis},
  pages={102748},
  year={2023},
  publisher={Elsevier}
}
~~~  
