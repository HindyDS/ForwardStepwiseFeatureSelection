#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import cross_val_score

class RecursiveFeatureSelector:
    def __init__(self):
        # number of trials as keys, best Subsets as values
        self.best_com = {}
        # number of trials as keys, best score as values
        self.best_score = {}
        # result
        self.trial_best = {}
        # store time spent for trials
        self.trials_time_spend = {}
        
    def trial(self, estimator, X, y, cv, scoring, max_round=None, fail_tolerance=None, least_gain=None, max_feats=None, start_from=None, pool=None, n_jobs=-1, n_digit=4):
        trial_start_time = time.time()
        n_trial = 1
        
        if fail_tolerance != None:    
            if fail_tolerance < 0 or isinstance(fail_tolerance, int) != True:
                return 'fail_tolerance(number of chances given Recursive Feature Selector to fail) must be positive integer.'
        if fail_tolerance == None:
            fail_tolerance = 1
            
        if max_round != None:
            if max_round < 0 or isinstance(max_round, int) != True:
                return 'max_round(maximum number of rounds given Recursive Feature Selector to search) must be positive integer.'    
        if max_round == None:
            max_round = np.inf
        
        if least_gain != None:
            if least_gain < 0 != True:
                return 'least_gain must be positive number.'
        
        if max_feats != None:
            if max_feats < 0 or isinstance(max_feats, int) != True:
                return 'max_feats(maximum number of features) must be positive integer.'
        if max_feats == None:
            max_feats = np.inf
            
        if start_from != None:
            if isinstance(start_from, list) != True:
                return 'start_from must be a list of features.'
            
        if n_digit != None:
            if n_digit < 0 or isinstance(n_digit, int) != True:
                return 'n_digit(number of digits) must be positive integer.'
        if n_digit == None:
            n_digit = 9   
            
        estimator_str = str(estimator).split('(')[0]
        print(f'Searching the best subset of features with {estimator_str}...')
        if start_from != None:
            print(f'Starting with {start_from}...')
            
        if start_from != None and type(start_from) != list:   
            print('start_from only accept list as argument.')
            return 
            
        if start_from != None:
            features2 =[]
            if type(start_from) == list:
                features = list(X.columns)
                for f in start_from:
                    features.remove(f)
                
                for feat in features:
                    features2.append(start_from + [feat])
            features = features2
            
        if start_from == None:    
            features = []
            for feature in X.columns:
                features.append([feature])
        
        while True:
            start_time = time.time()
            # features as keys, score as values
            feat_com = {}
            print(f'----------------------------------------------------------Trial {n_trial}----------------------------------------------------------')
            in_trial_count = 1
            # try out all features
            if n_trial == 1 and start_from == None:
                for feature in features:
                    cross_val_score_res = cross_val_score(estimator, X[feature], y, cv=cv, scoring=scoring, n_jobs=n_jobs)
                    score = round(cross_val_score_res.mean(), n_digit)
                    std = round(cross_val_score_res.std(), n_digit)
                    feat_com[feature[0]] = score
                    print(f'{in_trial_count}/{len(features)}: {feature}')
                    scoring_str = ' '.join(scoring.split('_')).title().replace('Neg', 'Negative').replace('Rand', 'Random').replace('Max', 'Maximum')
                    print(f'      {scoring_str}: {score}, Standard Deviation: {std}')
                    print(' ')
                    in_trial_count += 1

            if n_trial > 1 or start_from != None:
                for feature in features:
                    cross_val_score_res = cross_val_score(estimator, X[feature], y, cv=cv, scoring=scoring, n_jobs=n_jobs)
                    score = round(cross_val_score_res.mean(), n_digit)
                    std = round(cross_val_score_res.std(), n_digit)
                    feat_com[tuple(feature)] = score
                    print(f'{in_trial_count}/{len(features)}: {feature}')
                    scoring_str = ' '.join(scoring.split('_')).title().replace('Neg', 'Negative').replace('Rand', 'Random').replace('Max', 'Maximum')
                    print(f'      {scoring_str}: {score}, Standard Deviation: {std}')
                    print(' ')
                    in_trial_count += 1
                    
            # pick the and store trial best
            self.best_com[f'Trial {n_trial}'] = max(feat_com, key=feat_com.get)
            self.best_score[f'Trial {n_trial}'] = max(feat_com.values())

            # define the current trial best
            curr_trial_best = self.best_com[f'Trial {n_trial}']
            
            if n_trial == 1 and start_from == None:
                # features without the selected trial best
                features.remove([curr_trial_best])
                # generating new Subsets of features
                features = [[curr_trial_best]+[i][0] for i in features]
            
            if n_trial > 1 or start_from != None:
                curr_trial_best2 = list(self.best_com.values())
                features.remove(list(curr_trial_best2[n_trial-1]))
                if type(curr_trial_best2[n_trial-2]) == tuple:
                    
                    for feature in features:
                        for f in list(curr_trial_best2[n_trial-2]):
                            try:
                                feature.remove(f)  
                            except:
                                continue
                        
                if type(curr_trial_best2[n_trial-2]) == str:
                    for feature in features:
                        feature.remove(curr_trial_best2[n_trial-2])
                features2 = []
                for feature in features:
                    features2.append(list(curr_trial_best2[n_trial-1])+feature)
                    
                features = features2
            
            # for pool elimination
            if pool != None:
                # if new added feature in any of the subpool, remove it from the subpool
                for subpool in pool:
                    if curr_trial_best[-1] in subpool:
                        subpool.remove(curr_trial_best[-1])
                        # remove the rest of the features of the subpool from the subsets
                        for feature in features: 
                            for p in subpool:
                                try:
                                    feature.remove(p)
                                except:
                                    continue
                                    
                    # remove dups in nested list (features)
                    for feature in features:
                        index = []
                        for i in range(len(features)):
                            if feature == features[i]:
                                index.append(i)

                        index = index[-(features.count(feature) - 1):]

                        count = 0
                        for idx in index:
                            if features.count(feature) >1:
                                del features[idx - count]
                            count += 1

                    # remove unmatched length subset
                    index = []
                    for idx, feature in enumerate(features):
                        if len(feature) != n_trial + 1:
                            index.append(idx)

                    count = 0
                    for idx in index:
                        del features[idx - count]
                        count += 1
                
            # define keys to compare values
            curr_key = f'Trial {n_trial}'
            last_key = f'Trial {n_trial - 1}'
            
            if last_key != 'Trial 0':
                if least_gain == None:
                    # if fail to improve score, then take away one chance
                    if self.best_score[curr_key] < self.best_score[last_key]:
                        fail_tolerance = fail_tolerance - 1
                        print(f'Failed to improve {scoring_str}.')
                if least_gain != None:
                    # if fail to improve score by a certain percentage, then take away one chance
                    if (self.best_score[curr_key] - self.best_score[last_key])/self.best_score[last_key] < least_gain:
                        fail_tolerance = fail_tolerance - 1
                        print(f'Failed to improve {scoring_str} by {least_gain * 100}%.')
            
            print(f'Best Subset Found in Trial {n_trial}: ')
            if type(self.best_com[f'Trial {n_trial}']) == str:
                print('    ',self.best_com[f'Trial {n_trial}'])
                
            if type(self.best_com[f'Trial {n_trial}']) == tuple:
                print('    ',list(self.best_com[f'Trial {n_trial}']))
            print(' ')
            print(f'Best {scoring_str} of Trial {n_trial}: ')
            print('    ',self.best_score[f'Trial {n_trial}'])
            print(' ')
            self.trial_best[self.best_com[f'Trial {n_trial}']] = self.best_score[f'Trial {n_trial}']
            
            n_trial += 1
            max_round = max_round - 1
            
            end_time = time.time()
            self.trials_time_spend[f'Trial {n_trial - 1}'] = round(end_time - start_time, 2)
            print(f"Time Spent for Trial {n_trial - 1}: {round(end_time - start_time, 2)}(s)")
            print(' ')
            
            if fail_tolerance <= 0:            
                print('Fail tolerance exceeded.')
                print('Trial stops.')
                break
            if max_round <= 0:
                print('Round maximum reached.')
                print('Trial stops.')
                break
            if len(features) <= 0:
                print('All features subsets have been tried out.')
                break
            if max_feats == n_trial - 1:
                print(f'Top {max_feats} features have been selected.')
                break    
                
        best_com2 = {}
        temp_list = []
        for key, val in self.best_com.items():
            if type(val) == str:
                temp_list.append(val)
                best_com2[key] = temp_list
            else:
                best_com2[key] = list(val)

        self.best_com = best_com2
        # self.trial_best = list(max(self.trial_best, key=self.trial_best.get))
        self.summary = pd.DataFrame([self.best_com, self.best_score, self.trials_time_spend], 
                                    index=['Best Subset', f'Best {scoring_str}', 'Time Spent']).T
        self.best_subset = self.summary.sort_values(f'Best {scoring_str}', ascending=False).iloc[0, 0]
        self.best_score_all = max(self.summary.iloc[:, 1])
        
        # store the result
        print(f'--------------------------------------------------------Trial Summary--------------------------------------------------------')
        try:
            print(f'Best Subset Found: ')
            print('    ',self.best_subset)
            print(' ')
            print(f'Best {scoring_str}: ')
            print('    ',self.best_score_all)
            print(' ')
        except:
            n_trial = n_trial - 1
            print(f'Best Subset Found: ')
            print('    ',self.best_subset)
            print(' ')
            print(f'Best {scoring_str}: ')
            print('    ',self.best_score_all)
            print(' ')
            
        trial_end_time = time.time()
        print(f"Total Time Spent: {round(trial_end_time - trial_start_time, 2)}(s)")
        
        # visualizing the trials
        sns.set_theme()
        fig, ax = plt.subplots(figsize=(15, 6))  
        sns.lineplot(x=[i + 1 for i in range(len(self.best_com.keys()))], y=self.best_score.values())
        plt.axvline(x = np.argmax(list(self.best_score.values())) + 1, color='green', linewidth=2, linestyle='--')
        plt.ylabel(f'{scoring_str}')
        plt.xlabel('Subsets')
        plt.xticks(range(1,len(self.best_score.values()) + 1), list(self.best_com.values()), rotation=90)
        plt.title(f'Best {scoring_str} of each trial reached'.title())
        sns.despine();
        plt.show()
        print(f'---------------------------------------------End of Recursive Features Selection--------------------------------------------')
