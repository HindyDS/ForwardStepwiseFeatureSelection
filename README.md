# <img src="https://raw.githubusercontent.com/HindyDS/RecursiveFeatureSelector/main/logo/RFS%2010.5.2021.png" height="277">

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![PyPI version](https://badge.fury.io/py/RecursiveFeatureSelector.svg)](https://badge.fury.io/py/RecursiveFeatureSelector)
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

RecursiveFeatureSelector aims to select the best features or the subset of features in machine learning tasks according to corresponding score with other incredible packages like numpy, pandas and sklearn.

This package is inspired by: 
PyData DC 2016 | A Practical Guide to Dimensionality Reduction 
Vishal Patel
October 8, 2016

- **Examples:** https://github.com/HindyDS/RecursiveFeatureSelector/tree/main/RFS%20examples
- **Email:** hindy888@hotmail.com
- **Source code:** https://github.com/HindyDS/RecursiveFeatureSelector/blob/main/RecurrsiveFeatureSelector/__init__.py
- **Bug reports:** https://github.com/HindyDS/RecursiveFeatureSelector/issues

It requires at least six arguments to run:

- model: machine learning model
- X (array): features space
- y (array): target
- cv (int): number of folds in a (Stratified)KFold
- scoring (str): see https://scikit-learn.org/stable/modules/model_evaluation.html

- max_round (int): number of rounds that you wanted RFS to stop searching
- chances_to_fail (int): how many times RFS can fail to find better subset of features 
- jump_start (list): starting point for RFS to search, must be corresponds to the columns of X
- n_digit (int): Decimal places for scoring

If you have any ideas for this packge please don't hesitate to bring forward!
