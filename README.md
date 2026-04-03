# Project Introduction
**Title**: Gradual Machine Learning for Semi-supervised Medical Image Classification Via Evolutionary Feature Optimization, called **GML-EMT**. 
It is a non-i.i.d based paradigm for semi-supervised medical image classification, which gradually infers test images in a factor graph by integrating multiple high-quality evidence factors provided by evolutionary feature optimization. 
The framework of GML-EMT as shown in Figure 1.
![framework.png](framework.png)
**Figure 1**: Taking a 5\% labeling regime as an example, we utilize two pretrained models, fine-tune them using 5\% of the training data, and extract feature vectors from medical images and concatenate them. Then, we define two optimization sub-tasks (FET and FST) and apply EMT for collaborative optimization, refining the basic feature vectors from the perspectives of CCD and KNN based on different individual encoding schemes and evolutionary strategies. An information-sharing module is designed to facilitate EMT collaborative optimization. Finally, we extract CCD and KNN factors from the feature vectors of labeled images optimized by EMT for the test images. And then GML is used to fuse multiple optimized factors for gradual inference on the test images.
# Instruction Manual
**Install dependencies**: 

Python 3.8.19
PyTorch 2.0.1
CUDA 11.8
cuDNN 8700
certifi=2020.12.5
future 0.18.2,
joblib 1.1.0
llvmlite 0.37.0
numba 0.54.1
numpy 1.20.3
scikit-learn 1.0
scipy 1.7.1
threadpoolctl 3.0.0
wincertstore 0.2

## How to Run on Your Data
1. **Datasets download:**
   ISIC2018: https://www.kaggle.com/datasets/farjanakabirsamanta/skin-cancer-dataset
   
   NCT-CRC-HE-100K: https://www.kaggle.com/datasets/imrankhan77/nct-crc-he-100k

3. **Datasets split:**
   Randomly split the datasets into 70%/10%/20% for training/validation/testing. Further randomly select 5% and 20% subsets from the training set as labeled data, and treat the remainder as unlabeled data with a fixed random seed.

4. **Fine-tune the pre-trained models:**
   Fine-tune two pre-trained models (DenseNet-121 and WRN50-2) using the labeled data.

5. **Extract multi-view feature vectors:**
   Use the fine-tuned DenseNet-121 and WRN50-2 models to extract feature vectors from medical images and concatenate them as the basic feature vectors. Then, save the feature vectors and label information of the training/validation/test sets in .pkl format separately for subsequent method testing.

6. **Data loader:**
   Use the Dataloader module in the provided code to load the extracted feature vectors and label information.

7. **EMT optimization:**
   Run the EMT optimization algorithm to search for the optimal feature optimization model.

8. **Factor extraction:**
   Extract the CCD and KNN factors using the Extract_factors method and save them as .pkl files.

9. **Run GML:**
   Run example.py in the GML module.

More experimental details can be found in the provided code.
