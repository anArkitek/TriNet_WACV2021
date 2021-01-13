# A Vector-based Representation to Enhance Head Pose Estimation [WACV21]

## Citation
If you find the code useful, please consider citing our paper using the following BibTeX entry.

> @InProceedings{Cao_2021_WACV,
    author    = {Cao, Zhiwen and Chu, Zongcheng and Liu, Dongfang and Chen, Yingjie},
    title     = {A Vector-Based Representation to Enhance Head Pose Estimation},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2021},
    pages     = {1188-1197}
}

## Paper
https://openaccess.thecvf.com/content/WACV2021/html/Chu_A_Vector-Based_Representation_to_Enhance_Head_Pose_Estimation_WACV_2021_paper.html

## Abstract
This paper proposes to use the three vectors in a rotation matrix as the representation in head pose estimation and develops a new neural network based on the characteristic of such representation. We address two potential issues existed in current head pose estimation works: 1. Public datasets for head pose estimation use either Euler angles or quaternions to annotate data samples. However, both of these annotations have the issue of discontinuity and thus could result in some performance issues in neural network training. 2. Most research works report Mean Absolute Error (MAE) of Euler angles as the measurement of performance. We show that MAE may not reflect the actual behavior especially for the cases of profile views. To solve these two problems, we propose a new annotation method which uses three vectors to describe head poses and a new measurement Mean Absolute Error of Vectors (MAEV) to assess the performance. We also train a new neural network to predict the three vectors with the constraints of orthogonality. Our proposed method achieves state-of-the-art results on both AFLW2000 and BIWI datasets. Experiments show our vector-based annotation method can effectively reduce prediction errors for large pose angles.


## Platform
+ Keras
+ Tensorflow
+ Ubuntu

## Code

Our contributon focuses on the new representation of rotation and the new metric. We implement the experiments based on the structure of [**FSA-Net**](https://github.com/shamangary/FSA-Net). Please refer to their paper and code repo for detailed implementation. Notice we have made two modifications:
+ We replace their backbone of SSR-Net with ResNet
+ We modify their output layer for predicting three vectors.
