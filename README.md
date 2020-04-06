# NeuralNetwork_Charity_Donation

## Project Background

- Create a deep-learning neural network to analyze and classify the success of charitable donations in order to determine the future decisions of the company—only those projects likely to be a success will receive any future funding. 

## Dataset Information

A csv file containing more than 34,000 organizations that have received funding over the years. 

Columns metadata: 

- EIN and NAME—Identification columns **(irrelevant variables)**
- APPLICATION_TYPE—Charity Foundation application type **(irrelevant variables)**
- AFFILIATION—Affiliated sector of industry  **(Features)**
- CLASSIFICATION—Government organization classification  **(Features)**
- USE_CASE—Use case for funding  **(Features)**
- ORGANIZATION—Organization type **(Features)**
- STATUS—Active status   **(irrelevant variables)**
- INCOME_AMT—Income classification  **(Features)**
- SPECIAL_CONSIDERATIONS—Special consideration for application **(irrelevant variables)**
- ASK_AMT—Funding amount requested  **(Features)**

- IS_SUCCESSFUL—Was the money used effectively **(Target)**

### The model's structure

Total params: 8,903
Trainable params: 8,903
Non-trainable params: 0

-----------------------------------------------------------------------------

## Analysis Report

1. In this deep-learning neural ntework model, there are total three hidden layers with 84, 50 and 20 neurons respectively. The input shape was (42,) and output layer's neurons was 1 (binary_classification). The optimizer is adam and loss metrics setting is binary_crossentropy.

2. The model's predictive accuracy was under 75%, which means the model was not able to predict correctly whether or not a target company will be re-funded over 57% of the time.

3. In order to try and improve the model predictive performance, I tried to remove outliers and noisy in ASK_AMT variable by using 3 times of zscore method. In addition, removing irrelerant variables provided a few improvement. [remove_outliers.PNG](/Analysis/remove_outliers.PNG)


Despite of reducing dimensionality with PCA (Principal Component Analysis), the model's accuracy didn't significantly improve.

4. I tried to add a LeakyReLU layer to deep-learning model, but it didn't work as I expected. Furthermore, adding some regularizers on nauron's kernel still failed.

## Analysis Outcomes

The model's accuracy recording:

![accuracy.PNG](/Analysis/accuracy.PNG)