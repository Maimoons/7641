
<!-- PROJECT LOGO -->
<br />
<div align="center">
    <img src="logo.png" alt="Logo" width="80" height="80">
  </a>

<h3 align="center">Supervider Learning</h3>
  <p align="center">
    OMSCS 7641 - Assignment 1 
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


This contains code for the first Assignment in the course.
It implements 5 classifiers:
 * Decision Treesv- decision_tree.py
 * Boosted Decision Trees - boost.py
 * Neural Network - neural_network.py
 * SVM - svm.py
 * KNN - knn.py
<p align="right">(<a href="#readme-top">back to top</a>)</p>



### Built With

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

The list of requirements for this repo is in requirements.txt so pip install each e.g
* pip
  ```sh
  pip install numpy
  ```

<!-- USAGE EXAMPLES -->
## Usage
There are two datasets used in thisa project.
* Dataset 0 - Titanic dataset from Kagg;e
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


