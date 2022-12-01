
<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Machine Learning</h3>
  <p align="center">
    OMSCS 7641 
  </p>
</div>



<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project
Github Repo - https://github.com/Maimoons/7641/blob/master/README.md

Project 4: Reinforcement Learning
The folder: Project 4 contains code for the forth assignment in the course.
It implements 3 different RL algorithms on 2 different MDP:
 * Policy Iteration- policy_iteration.py
 * Value Iteration - value_iteration.py
 * Q Learning - q_learn.py
 * Experiments - experiments.py
<p align="right">(<a href="#readme-top">back to top</a>)</p>

Project 3: Unsupervised Learning

The folder: Project 3 contains code for the third assignment in the course.
It implements 2 different clustering problems and 4 dimensionality reduction problems:
 * K Means- kmm.py
 * Expectation Maximization - em.py
 * PCA - pca.py
 * IPA - ipa.py
 * Gaussian Random Mixture - grp.py
 * Random Forest - rf.py
<p align="right">(<a href="#readme-top">back to top</a>)</p>

Project 2: Randomized Optimization

The folder: Project 2 contains code for the second assignment in the course.
It implements 4 different problems:
 * Neural Network- neural_network.py
 * Knapsack - knapsack.py
 * Traveling Salesman - tsp.py
 * Continuous Peaks - continuos_peaks.py
<p align="right">(<a href="#readme-top">back to top</a>)</p>


Project 1: Supervised Learning

The folder: Project 1 contains code for the first assignment in the course.
It implements 5 classifiers:
 * Decision Trees- decision_tree.py
 * Boosted Decision Trees - boost.py
 * Neural Network - neural_network.py
 * SVM - svm.py
 * KNN - knn.py
<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With
Project 4 (the following on top of the dependencies from project 1)
* hiive.mdptoolbox
* mdptoolbox
* openaigym

Project 3 (the following on top of the dependencies from project 1)
* sklears.clusters
* sklearn.metrics
* sklearn.manifolds
* sklearn.mixture
* sklearn.decomposition

Project 2 (the following on top of the dependencies from project 1)
* MLrose --hiive

Project 1
* Numpy
* Pandas
* SkLearn
* Matplotlib

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- GETTING STARTED -->
## Getting Started

Follow the commands to setup and run the experiments:

### Installation

1. Clone the repo
   ```sh
   git clone https://github.com/Maimoons/7641
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


### Prerequisites

The list of requirements for this repo is in requirements.txt as well as defined above so pip install each e.g
* pip
  ```sh
  pip install numpy
  ```

<!-- USAGE EXAMPLES -->
## Usage

<h3 align="center">Project 4</h3>

There are two problems used in this project.
* Grid problem
* Non grid problem

The configuration for both is defined in the experiments file - 

Running  the experiments
   ```sh
   python experiments.py 0
   ```

The output graphs are produced in the images folder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<h3 align="center">Project 3</h3>

There are two datasets used in this project.
* Dataset 0 - Titanic dataset from Kaggle
* Dataset 1 - Breast cancer dataset from sklearn

The dataset index needs to be passed in to run the experiments - 

Running  the experiments
   ```sh
   python run_experiments.py 0
   ```

The output graphs are produced in the images folder.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<h3 align="center"> Project 2 </h3>

The dataset used for the Neural Network problem - 
* Dataset 0 - Titanic dataset from Kaggle

Running  the classifier
   ```sh
   python neural_network.py
   ```

Running any of the second half of optimization problems

   ```sh
   python tsp.py
   python knapsack.py
   python continuos_peaks.py
   ```

The output graphs are produced in the images folder.

To plot training and testing time bar graphs for Neural Network - 

   ```sh
   python base.py
   ```

<br>
<br>
   
   
   
   
<h3 align="center">Project 1</h3>

There are two datasets used in this project.
* Dataset 0 - Titanic dataset from Kaggle
* Dataset 1 - Breast cancer dataset from sklearn

The dataset index needs to be passed in to whatever classifier is being run. For example - 

Running  the classifier
   ```sh
   python decision_tree.py 0
   ```

The above calls both the training and testing and saves the trained model. The output graphs are produced in the images folder.

To plot training and testing time bar graphs - 
Running  the classifier
   ```sh
   python base.py
   ```

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- CONTACT -->
## Contact

Your Name -  msiddiqui61@gatech.edu
<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [https://github.com/othneildrew/Best-README-Template/blob/master/BLANK_README.md](Readme Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>



<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->


