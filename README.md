# Advanced Geospatial Analysis for Land Cover Detection using Transfer Learning
Machine Learning model to classify various types of land cover using high-resolution satellite imagery


## Table of Contents
- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [How To Use](#how-to-use)
- [Outputs](#outputs)


## Introduction

This project involves the design and implementation of a machine learning model to classify various types of land cover using high-resolution satellite imagery and Transfer Learning. Transfer learning is a technique where a model developed for one task is reused as the starting point for a model on a second task. It leverages pre-trained models, which have been trained on large datasets and extensive computational resources, to solve new but related problems more efficiently. This approach is particularly useful when the new task has limited data, as the pre-trained model's learned features and representations can significantly enhance performance and reduce training time. For instance, a model like ResNet-50, which has been trained on the ImageNet dataset, can be fine-tuned for specific tasks such as land cover classification using high-resolution satellite imagery, by adjusting its weights to adapt to the new dataset.

### Transfer Learning
<img src="https://github.com/dineshg20897/lulc_transfer_learning/blob/main/assets/Transfer.png?raw=true" width="800"><br><br>

### RESNET 50
<img src="https://github.com/dineshg20897/lulc_transfer_learning/blob/main/assets/Resnet.png?raw=true" width="800"><br><br>


## Objective

The goal is to automatically assign labels that describe the physical land type or land usage shown in these image patches. To achieve this, an image patch is fed into a classifier, such as a neural network, which then outputs the corresponding class for the image patch.

The satellite captures 13 spectral bands. Among these, bands B01, B09, and B10 are used for correcting atmospheric effects (e.g., aerosols, cirrus, or water vapor). The remaining bands are primarily used to identify and monitor land use and land cover classes. Each satellite is designed to provide imagery for at least seven years with a spatial resolution of up to 10 meters per pixel.

To enhance the quality of the image patches, satellite images with low cloud cover are selected. The European Space Agency (ESA) provides a cloud level value for each satellite image, which helps in quickly selecting images with minimal cloud coverage over the land scene. Additionally, the option to generate a cloud mask further aids in ensuring valuable image patches.


## Dataset

The EuroSAT dataset is a widely used benchmark for land cover classification tasks in the field of remote sensing. It consists of high-resolution satellite imagery collected by the Sentinel-2 satellite, part of the European Space Agency's (ESA) Copernicus Earth Observation program. The dataset is designed to facilitate the development and evaluation of machine learning models for classifying different types of land cover.

<img src="https://github.com/dineshg20897/lulc_transfer_learning/blob/main/assets/Dataset.png?raw=true" width="800"><br><br>

### Key Features of the EuroSAT Dataset

1. **Spectral Bands**: The dataset includes images with 13 spectral bands, ranging from visible to infrared wavelengths. These bands capture a wide range of information about the Earth's surface, allowing for detailed analysis of various land cover types.

2. **Image Resolution**: The spatial resolution of the images is up to 10 meters per pixel for certain bands, providing detailed and high-quality imagery suitable for fine-grained land cover classification.

3. **Classes**: The dataset contains images labeled with 10 different land cover classes.

4. **Size**: The dataset includes over 27,000 labeled images, making it a comprehensive resource for training and testing machine learning models.

5. **Preprocessing**: To ensure the quality of the dataset, images with high cloud cover are excluded. Additionally, ESA provides cloud cover values and the ability to generate cloud masks to filter out unwanted cloud-covered areas.

6. **Access**: The EuroSAT dataset is publicly available and can be accessed through various platforms, including:
      - [EuroSAT GitHub Repository](https://github.com/phelber/eurosat)
      - [European Space Agency (ESA) Copernicus Open Access Hub](https://scihub.copernicus.eu/dhus)


## How to use

1. Ensure the below-listed packages are installed.
    - `NumPy`
    - `matplotlib`
    - `Tensorflow`
    - `Keras`
    - `Rasterio`
    - `scikit-learn`
    - `seaborn`
    - `PIL`
    - `os`
    - `pandas`
2. Download `Land_Cover_Classification.ipynb` jupyter notebook from this repository.
3. Download the dataset and place it in the `data` directory.
4. Execute the notebook from start to finish in one go. The transfer learning models are downloaded automatically. 
5. Experiment with different hyperparameters – longer training would yield better results.


## Outputs

The Images below show the learning curves during the training phase

<img src="https://github.com/dineshg20897/lulc_transfer_learning/blob/main/assets/Loss.png?raw=true" width="800"><br><br>
<img src="https://github.com/dineshg20897/lulc_transfer_learning/blob/main/assets/Learning.png?raw=true" width="800"><br><br>

And this is the confusion matrix

<img src="https://github.com/dineshg20897/lulc_transfer_learning/blob/main/assets/Conf.png?raw=true" width="800"><br><br>
