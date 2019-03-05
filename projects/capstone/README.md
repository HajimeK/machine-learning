# Machine Learning Engineer Nanodegree
## Specializations
## Project: Capstone Proposal and Capstone Project

## Proposal

- proposal.pdf (Generaded from *capstone_proposal.md)
- Review: https://review.udacity.com/#!/reviews/1726191
- data set
  - (Training  ) data/aps_failure_training_set_processed_8bit.csv
  - (Evaluation) data/aps_failure_test_set_processed_8bit.csv

## Project

- capstone.ipynb
- report.pdf


## Libraries Used (Imports)


# Import libraries necessary for this project

- Python 3.6
- conda 4.6.7
- OS: Ubuntu 16.04 / Mac OS 10.13.6

Most of the tools used in the capston.ipynb have been installed with Anaconda distribution (https://www.anaconda.com/distribution/).
Some packages need to be installed manually with the following commands.

- conda install seaborn
- conda install -c conda-forge xgboost
- pip install graphviz
- conda install -y graphviz
- pip install graphviz
- apt-get install -y libltdl7

```
import numpy as np
import pandas as pd
from matplotlib import pyplot
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display # Allows the use of display() for DataFrames
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics.scorer import make_scorer, accuracy_score, recall_score, roc_auc_score, r2_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier


import xgboost as xgb
from xgboost import plot_tree
from xgboost import plot_importance
```