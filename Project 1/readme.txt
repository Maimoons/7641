About The Project
Github Repo - https://github.com/Maimoons/7641/blob/master/README.md

This contains code for the first Assignment in the course. It implements 5 classifiers:

Decision Trees- decision_tree.py
Boosted Decision Trees - boost.py
Neural Network - neaural_network.py
SVM - svm.py
KNN - knn.py


Built With
Numpy
Pandas
SkLearn
Matplotlib
cPickle
logging

Getting Started

Installation
Clone the repo
git clone https://github.com/Maimoons/7641

Prerequisites
The list of requirements for this repo is in requirements.txt
pip install the requirements.

Usage
There are two datasets used in this project.

Dataset 0 - Titanic dataset from Kaggle
Dataset 1 - Breast cancer dataset from sklearn
The dataset index needs to be passed in to whatever classifier is being run. 
For example - Running the classifier

python decision_tree.py 0

The above calls both the training and testing and saves the trained model. The output graphs are produced in the images folder.


For the final time graphs run base.py
Python base.py

