# ForwardStepwiseFeatureSelection

[![Open Source Love](https://badges.frapsoft.com/os/v2/open-source.svg?v=103)](https://github.com/ellerbrock/open-source-badges/)
[![PyPI version](https://badge.fury.io/py/ForwardStepwiseFeatureSelection.svg)](https://badge.fury.io/py/ForwardStepwiseFeatureSelection)
[![MIT Licence](https://badges.frapsoft.com/os/mit/mit.svg?v=103)](https://opensource.org/licenses/mit-license.php)

ForwardStepwiseFeatureSelection aims to select the best features or the subset of features in machine learning tasks according to corresponding score with other incredible packages like numpy, pandas and sklearn.

## Quick Start
	# Install ForwardStepwiseFeatureSelection
	!pip install ForwardStepwiseFeatureSelection
	
## Quick Example
	# Import dependenices
	from ForwardStepwiseFeatureSelection import ForwardStepwiseFeatureSelection
	from sklearn.model_selection import train_test_split
	from sklearn.preprocessing import StandardScaler
	from sklearn.ensemble import RandomForestRegressor
	import pandas as pd

	# Read dataframe
	insurance = pd.read_csv('insurance.csv')

	# Label Encoding
	for col in ['sex', 'smoker', 'region']:
	    insurance[col].replace(insurance[col].unique(), range(insurance[col].nunique()), inplace=True)

	X = insurance.drop('charges', axis=1)
	y = insurance['charges']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

	# Scale our data
	scaler = StandardScaler()
	X_train, X_test = (pd.DataFrame(scaler.fit_transform(df), columns=df.columns) for df in [X_train, X_test])
	X_train, X_test, y_train, y_test = train_test_split(X, y)

	# Instantiate the estimator
	rfc = RandomForestRegressor()

	# Instantiate ForwardStepwiseFeatureSelection
	fsfs = ForwardStepwiseFeatureSelection(estimators=rfc, cv=3, scoring='neg_mean_absolute_error', mode=None, verbose=1, tolerance=3)

	# Start feature selection
	fsfs.fit(X_train, y_train)

	print(fsfs.best_subsets)
	
	>> {'RandomForestRegressor': ['smoker', 'age', 'bmi', 'children', 'region']}

This package is inspired by: 
PyData DC 2016 | A Practical Guide to Dimensionality Reduction 
Vishal Patel
October 8, 2016

- **Examples:** https://github.com/HindyDS/ForwardStepwiseFeatureSelection/tree/main/examples
- **Email:** hindy888@hotmail.com
- **Source code:** https://github.com/HindyDS/ForwardStepwiseFeatureSelection/tree/main/ForwardStepwiseFeatureSelection
- **Bug reports:** https://github.com/HindyDS/ForwardStepwiseFeatureSelection/issues

It requires at least six arguments to run:

- estimators: machine learning model
- X (array): features space
- y (array): target
- cv (int): number of folds in a (Stratified)KFold
- scoring (str): see https://scikit-learn.org/stable/modules/model_evaluation.html

Optional arguments:
- mode (string): None or 'ts'. If 'ts' (Time Series) than it will change to walk forward cross validation. 
- max_trial (int): number of trials that you wanted FSFS to stop searching
- tolerance (int): how many times FSFS can fail to find better subset of features 
- least_gain (int): threshold of scoring metrics gain in fraction 
- max_feats (int): maximum number of features
- prior (list): starting point for FSFS to search, must be corresponds to the columns of X
- exclusions (nested list): if the new selected feature is in one of the particular subpool 
		    (list in the nested list), then the features in that particular subpool with no 			    longer be avalible to form any new subset in the following trials
- n_jobs (int): Number of jobs to run in parallel.
- n_digit (int): Decimal places for scoring
- verbose (int): Level of verbosity of FSFS

If you have any ideas for this packge please don't hesitate to bring forward!
