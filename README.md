# Bleeding-Detection

The aim of this project is to classify a given image, recorded by Wireless Capsule Endoscopy, as containing bleeding or not. For this task, methods based on [Sparse Coding](https://hal.archives-ouvertes.fr/hal-01145892/document), [SIFT](https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf) and [Local Binary Pattern](http://www.mediateam.oulu.fi/publications/pdf/94.pdf) are developed. 

- Approach 1: Sparse coding is used for intensity based features.
- Approach 2: Sparse coding is combined with Dense SIFT descriptors and LBP features.
- Approach 3: Similar to Approach 2 but sparse coding is combined with SIFT keypoints descriptors and LBP features.
