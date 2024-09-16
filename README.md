
<div align="center">

![logo](https://github.com/ShailadhShinde/CNN/blob/main/assets/header.png)  
<h1 align="center"><strong>Statoil/C-CORE Iceberg Classifier Challenge
 <h6 align="center">A Featured Prediction (CV) project</h6></strong></h1>

![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)

</div>
This project focuses on:

- Image Classification using CNN
- Demonstration of Transfer learning using VGG16
- Implementaion using CNN Architectures

#### -- Project Status: [Completed]

#### -- iceberg.py / iceberg.ipynb - Contains code for the project

----

## [📚 Project Documentation 📚](http://smp.readthedocs.io/)

### 📋 Table of Contents
- [Overview](#overview)
  - [About the data](#atd)
  - [Preprocessing](#pp)
  - [Things I tried but did not work](#TT)
  - [Evaluation](#eval)
  - [Architecture](#arch)
- [Demo](#demo)
- [Results](#results)
- [Getting Started](#gs)
  - [Prerequisites](#pr)


###  📌 Project Overview  <a name="overview"></a>

This project is a notebook which will predict whether an image contains a ship or an iceberg. The labels are provided by human experts and geographic knowledge on the target. All the images are 75x75 images with two bands.
- ### About the data <a name="atd"></a>

Sentinet -1 sat is at about 680 Km above earth. Sending pulses of signals at a particular angle of incidence and then recoding it back. Basically those reflected signals are called backscatter. The data we have been given is backscatter coefficient which is the conventional form of backscatter coefficient given by:

`σo(dB)=βo(dB)+10log10[sin(ip)/sin(ic)]`

where

- ip=is angle of incidence for a particular pixel
- 'ic ' is angle of incidence for center of the image
- K =constant.
  
We have been given `σo` directly in the data.

Now coming to the features of σo
Basically σo varies with the surface on which the signal is scattered from. For example, for a particular angle of incidence, it varies like:

WATER........... SETTLEMENTS........ AGRICULTURE........... BARREN........

1.HH: -27.001 ................ 2.70252 ................. -12.7952 ................ -17.25790909

2.HV: -28.035 ................ -20.2665 .................. -21.4471 ................. -20.019

As you can see, the HH component varies a lot but HV doesn't. I don't have the data for scatter from ship, but being a metal object, it should vary differently as compared to ice object.

#### WTF is HH HV?

Ok, so this Sentinal Settalite is equivalent to RISTSAT(an Indian remote sensing Sat) and they only Transmit pings in H polarization, AND NOT IN V polarization. Those H-pings gets scattered, objects change their polarization and returns as a mix of H and V. Since Sentinel has only H-transmitter, return signals are of the form of HH and HV only. Don't ask why VV is not given(because Sentinel don't have V-ping transmitter).

Now coming to features, for the purpose of this demo code, I am extracting all two bands and taking avg of them as 3rd channel to create a 3-channel RGB equivalent.


#### Data fields

`train.json`,` test.json`

The files consist of a list of images, and for each image, you can find the following fields:

- id - the id of the image 
- band_1, band_2 - the flattened image data. Each band has 75x75 pixel values in the list, so the list has 5625 elements. Note that these values are not the normal non-negative integers in image files since they have physical meanings - these are float numbers with unit being dB. Band 1 and Band 2 are signals characterized by radar backscatter produced from different polarizations at a particular incidence angle. The polarizations correspond to HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically). More background on the satellite imagery can be found here.
inc_angle - the incidence angle of which the image was taken. Note that this field has missing data marked as "na", and those images with "na" incidence angles are all in the training data to prevent leakage.
- is_iceberg - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship. This field only exists in train.json.

`The train data` contains time series of the stores and the product families combination. The sales column gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).The onpromotion column gives the total number of items in a product family that were being promoted at a store at a given date.

- ### Preprocessing  <a name="pp"></a>
  - Due to the relatively small train set I used Image augmentation.For augmentation I used flip, flop, zoom, rotation and shift.
  - Added third channel as band3 (band1 + band2)/2.
Replaced NA inc_angle with 0

- ### Things I tried but did not work  <a name="TT"></a>

  - preprocessing in ['mean_std', 'minmax', 'raw']
  - cnn_type in ['simple','double']
  - conv_layers in [[64,128,128,64], [32, 64, 128,128], [64, 128, 256]]
  - FE in filters on images ['gaussian','laplace',etc]
  - transfer learning ['Vgg16']
  - various custom cnn architecture

- ### Evaluation  <a name="eval"></a>
  The evaluation metric used is Log-Loss (Cross -Entropy)
   ![logloss](https://github.com/ShailadhShinde/CNN/blob/main/assets/eval.JPG)


- ### Architecture <a name="arch"></a>

Architecture used 

    Input Shape: (75, 75, 3)

    Conv2D (64 filters, 3x3 kernel, ReLU) 
    Conv2D (64 filters, 3x3 kernel, ReLU)
    Conv2D (64 filters, 3x3 kernel, ReLU)
    MaxPooling2D (3x3 pool, strides 2x2)

    Conv2D (128 filters, 3x3 kernel, ReLU)
    Conv2D (128 filters, 3x3 kernel, ReLU)
    Conv2D (128 filters, 3x3 kernel, ReLU)
    MaxPooling2D (2x2 pool, strides 2x2)

    Conv2D (128 filters, 3x3 kernel, ReLU)
    MaxPooling2D (2x2 pool, strides 2x2)

    Conv2D (256 filters, 3x3 kernel, ReLU)
    MaxPooling2D (2x2 pool, strides 2x2)

    Flatten

    Dense (1024 units, ReLU)
    Dropout (0.4)

    Dense (512 units, ReLU)
    Dropout (0.2)

    Dense (1 unit, Sigmoid)
    


----

## ✨ Demo <a name="demo"></a>

Inputs

   <p align="center">
  <img width="60%" height ="40%"  src="https://github.com/ShailadhShinde/CNN/blob/main/assets/1.JPG">
 </p>
   <p align="center">
  <img width="60%" height ="300"  src="https://github.com/ShailadhShinde/CNN/blob/main/assets/2.JPG">
 </p>
   <p align="center">
  <img width="60%" height ="400"  src="https://github.com/ShailadhShinde/CNN/blob/main/assets/ship.png">
 </p>
   <p align="center">
  <img width="60%" height ="400"  src="https://github.com/ShailadhShinde/CNN/blob/main/assets/iceberg.png">
 </p>
 

----
## 💫 Results <a name="results"></a>

  The top score of the competiton was 0.08227 usign 100s of models.
  
  Got a score of 0.15943 using only a single model 😄
  
   <p align="center">
  <img width="60%" src="https://github.com/ShailadhShinde/CNN/blob/main/assets/score.jpg">
 </p>

  
---

## 🚀 Getting Started <a name="gs"></a>

### ✅ Prerequisites <a name="pr"></a>
 
 - <b>Dataset prerequisite for training</b>:
 
 Before starting to train a model, make sure to download the dataset from <a href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data" target="_blank">here </a> or add it to your notebook
 ### 🐳 Setting up and Running the project

 Just download/copy the files `iceberg.py / iceberg.ipynb ` and `EDA.ipynb / EDA.py ` and run them

  
