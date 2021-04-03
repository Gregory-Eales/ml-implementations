<h1 align="center"> Linear Regression </h1>

<h4 align="center"> Reimplementation of Linear Regression </h4>

<p align="center">
  <img src="https://img.shields.io/badge/Python-v3.6+-blue.svg">
  <img src="https://img.shields.io/badge/Numpy-v1.16.4-orange.svg">
  <img src="https://img.shields.io/badge/Status-Complete-green.svg">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg">
</p>

<p align="center">
  <a href="#About">About</a> •
  <a href="#Requirements">Requirements</a> •
  <a href="#Algorithm">Algorithm</a> •
  <a href="#Environment">Environment</a> •
  <a href="#Training">Training</a> •
  <a href="#Results">Results</a> •
  <a href="#Sources">Sources</a>
</p>

## About:
Implementation of linear regression trained to perform binary classification.

## Requirements:
- Numpy v1.16.4
- Matplotlib
- Sklearn


## Algorithm:
Linear regression is a line fitting algorithm that updates its parameters using gradient descent. In this implementation, linear regression is used as a binary classifier to predict if a data sample is from a particular classification.

## Data:
The data used is generated randomly to form two slightly overlapping gaussian distributions. This provides a perfect test case for a simple binary classifier.

<p align="center">
  <img width="800" src="https://github.com/Gregory-Eales/ML-Reimplementations/blob/master/Linear-Regression/img/class_data.png">
</p>

## Training:
The model is trained using gradient descent with out batching and a learning rate of 0.001 with 10,000 iterations.

<p align="center">
  <img width="800" src="https://github.com/Gregory-Eales/ML-Reimplementations/blob/master/Linear-Regression/img/loss_per_iter.png">
</p>


## Results:
The model is able to achieve a very convincing decision boundary, clearly dividing the two class distributions and is able to correctly classify data with around 97% accuracy. 

<p align="center">
  <img width="800" src="https://github.com/Gregory-Eales/ML-Reimplementations/blob/master/Linear-Regression/img/accuracy_per_epoch.png">
</p>

<p align="center">
  <img width="800" src="https://github.com/Gregory-Eales/ML-Reimplementations/blob/master/Linear-Regression/img/decision_boundry.png">
</p>

## Sources:
- show sources

## Meta:

Gregory Eales – [@GregoryHamE](https://twitter.com/GregoryHamE) – gregory.hamilton.e@gmail.com

Distributed under the MIT license. See ``LICENSE`` for more information.

[https://github.com/Gregory-Eales](https://github.com/Gregory-Eales)
