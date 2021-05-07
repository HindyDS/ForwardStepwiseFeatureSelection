# <img src="https://raw.githubusercontent.com/HindyDS/RecurrsiveFeatureSelector/main/logo/RFS2.png" height="90">

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![PyPI version](https://badge.fury.io/gh/HindyDS%2FRecurrsiveFeatureSelector.svg)](https://pypi.org/project/RecurrsiveFeatureSelector/)
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

RecurrsiveFeatureSelector aims to select the best features in machine learning tasks according to corresponding score with other incredible packages like numpy, pandas and sklearn.

This package is inspired by: 
PyData DC 2016 | A Practical Guide to Dimensionality Reduction 
Vishal Patel
October 8, 2016

- **Examples:** https://github.com/HindyDS/RecurrsiveFeatureSelector/tree/main/examples
- **Email:** hindy888@hotmail.com
- **Source code:** https://github.com/HindyDS/RecurrsiveFeatureSelector/tree/main/RecurrsiveFeatureSelector
- **Bug reports:** https://github.com/HindyDS/RecurrsiveFeatureSelector/issues

It requires at least six arguments to run:

- model: machine learning model
- X (array): features space
- y (array): target
- cv (int): number of folds in a (Stratified)KFold
- task (str): 'classification'/'regression'
- scoring (str): see https://scikit-learn.org/stable/modules/model_evaluation.html

- max_round (int): number of rounds that you wanted RFS to stop searching
- chances_to_fail (int): how many times RFS can fail to find better subset of features 
- jump_start (list): starting point for RFS to search, must be corresponds to the columns of X
- n_digit (int): Decimal places for scoring

If you have any ideas for this packge please don't hesitate to bring forward!
