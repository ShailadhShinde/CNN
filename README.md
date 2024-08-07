
<div align="center">

![logo](https://github.com/ShailadhShinde/Time_series/blob/main/assets/header.png)  
<h1 align="center"><strong>Statoil/C-CORE Iceberg Classifier Challenge
 <h6 align="center">A Featured Prediction (CV) project</h6></strong></h1>

![Python - Version](https://img.shields.io/badge/PYTHON-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)

</div>
This project focuses on:

- Exploratory Data Analysis
- 
-
-

#### -- Project Status: [Completed]

#### -- time_series.py / time_Series.ipynb - Contains code for the project

#### -- eda-time-series.ipynb / EDA.py - Exploratory Data Analysis [Click to view](https://www.kaggle.com/code/shailadh/eda-time-series?scriptVersionId=190759981)

----

## [📚 Project Documentation 📚](http://smp.readthedocs.io/)

### 📋 Table of Contents
- [Overview](#overview)
  - [About the dataset](#atd)
  - [Sample Selection](#ss)
  - [Preprocessing](#pp)
  - [Feature Engineering](#fe)
  - [Evaluation](#eval)
  - [Model](#model)
- [Results](#results)
- [Getting Started](#gs)
  - [Prerequisites](#pr)


###  📌 Project Overview  <a name="overview"></a>

This project is a Notebook about time series forcasting for store sales.The purpose is to predict sales for 1000s of products sold at favourite stores located in South America’s west coast Ecuador. [Click here for more INFO](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data)

- ### About the dataset  <a name="atd"></a>

Intro about the Data.¶
Sentinet -1 sat is at about 680 Km above earth. Sending pulses of signals at a particular angle of incidence and then recoding it back. Basically those reflected signals are called backscatter. The data we have been given is backscatter coefficient which is the conventional form of backscatter coefficient given by:

σo(dB)=βo(dB)+10log10[sin(ip)/sin(ic)]
where

ip=is angle of incidence for a particular pixel
'ic ' is angle of incidence for center of the image
K =constant.
We have been given σo
 directly in the data.

Now coming to the features of σo
Basically σo varies with the surface on which the signal is scattered from. For example, for a particular angle of incidence, it varies like:

WATER........... SETTLEMENTS........ AGRICULTURE........... BARREN........
1.HH: -27.001 ................ 2.70252 ................. -12.7952 ................ -17.25790909

2.HV: -28.035 ................ -20.2665 .................. -21.4471 ................. -20.019

As you can see, the HH component varies a lot but HV doesn't. I don't have the data for scatter from ship, but being a metal object, it should vary differently as compared to ice object.

WTF is HH HV?
Ok, so this Sentinal Settalite is equivalent to RISTSAT(an Indian remote sensing Sat) and they only Transmit pings in H polarization, AND NOT IN V polarization. Those H-pings gets scattered, objects change their polarization and returns as a mix of H and V. Since Sentinel has only H-transmitter, return signals are of the form of HH and HV only. Don't ask why VV is not given(because Sentinel don't have V-ping transmitter).

Now coming to features, for the purpose of this demo code, I am extracting all two bands and taking avg of them as 3rd channel to create a 3-channel RGB equivalent.


 you will predict whether an image contains a ship or an iceberg. The labels are provided by human experts and geographic knowledge on the target. All the images are 75x75 images with two bands.

Data fields
train.json, test.json
The data (train.json, test.json) is presented in json format.
The files consist of a list of images, and for each image, you can find the following fields:

id - the id of the image
band_1, band_2 - the flattened image data. Each band has 75x75 pixel values in the list, so the list has 5625 elements. Note that these values are not the normal non-negative integers in image files since they have physical meanings - these are float numbers with unit being dB. Band 1 and Band 2 are signals characterized by radar backscatter produced from different polarizations at a particular incidence angle. The polarizations correspond to HH (transmit/receive horizontally) and HV (transmit horizontally and receive vertically). More background on the satellite imagery can be found here.
inc_angle - the incidence angle of which the image was taken. Note that this field has missing data marked as "na", and those images with "na" incidence angles are all in the training data to prevent leakage.
is_iceberg - the target variable, set to 1 if it is an iceberg, and 0 if it is a ship. This field only exists in train.json.
  `The train data` contains time series of the stores and the product families combination. The sales column gives the total sales for a product family at a particular store at a given date. Fractional values are possible since products can be sold in fractional units (1.5 kg of cheese, for instance, as opposed to 1 bag of chips).The onpromotion column gives the total number of items in a product family that were being promoted at a store at a given date.

- ### Preprocessing  <a name="pp"></a>
  Filled missing or negtive promotion and target values with 0.

- ### Feature Engineering  <a name="fe"></a>
 1. #### Basic features
    * category features: store, family, city, state, type
    * promotion

 
- ### Things I tried but did not work for me  <a name="eval"></a>

  - Transfer Learning (VGG 16)
  - Custom model 1 
  
- ### Evaluation  <a name="eval"></a>
  The evaluation metric used is Root Mean Squared Logarithmic Error. RMSLE = $\sqrt{\frac{1}{n} \sum_{i=1}^n (\log(p_i + 1) - \log(a_i+1))^2 }$

- ### Model <a name="model"></a>

  

----

## 💫 Results <a name="results"></a>

  Got a good score resulting in top 1 % of the kaggle leader board
  
   <p align="center">
  <img width="60%" src="https://github.com/ShailadhShinde/Time_series/blob/main/assets/score.JPG">
 </p>

  
---

## 🚀 Getting Started <a name="gs"></a>

### ✅ Prerequisites <a name="pr"></a>
 
 - <b>Dataset prerequisite for training</b>:
 
 Before starting to train a model, make sure to download the dataset from <a href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data" target="_blank">here </a> or add it to your notebook
 ### 🐳 Setting up and Running the project

 Just download/copy the files `time_series.py / time_Series.ipynb ` and `EDA.ipynb / EDA.py ` and run them

  
