#!/usr/bin/env python
# coding: utf-8

# In[ ]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.model_selection import cross_val_score

class RecurrsiveFeatureSelector:
    def __init__(self):
        # number of trials as keys, best combinations as values
        self.best_com = {}
        # number of trials as keys, best score as values
        self.best_score = {}
        # result
        self.trial_best = {}
        # store time spent for trials
        self.trials_time_spend = {}
        
    def trial(self, model, X, y, cv, task, scoring, max_round=None, chances_to_fail=None, jump_start=None, n_digit=4):
        trial_start_time = time.time()
        n_trial = 1
        if chances_to_fail == None:
            chances_to_fail = 1
        if max_round == None:
            max_round = np.inf
        if task == 'classification':
            for s in ['accuracy', 'balanced_accuracy', 'top_k_accuracy', 'average_precision', 'neg_brier_score', 'f1',
                      'f1_micro', 'f1_macro', 'f1_weighted', 'f1_samples', 'neg_log_loss', 'precision', 'recall', 'jaccard', 
                      'roc_auc', 'roc_auc_ovr', 'roc_auc_ovo', 'roc_auc_ovr_weighted', 'roc_auc_ovo_weighted']:
                if scoring == s:
                    scoring = s
                    continue
        if task == 'regression':
            for s in ['explained_variance', 'max_error', 'neg_mean_absolute_error', 'neg_mean_squared_error', 
                      'neg_root_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'r2', 
                      'neg_mean_poisson_deviance', 'neg_mean_gamma_deviance', 'neg_mean_absolute_percentage_error', 
                      'mean_absolute_error', 'mean_squared_error', 'root_mean_squared_error']:
                if scoring == s:
                    scoring = s
                    continue
            
        print('Trial Started:')
        model_str = str(model).split('(')[0]
        print(f'Searching the best subset of features with {model_str}...')
            
        if jump_start != None and type(jump_start) != list:   
            print('jump_start only accept list as argument.')
            return 
            
        if jump_start != None:
            features2 =[]
            if type(jump_start) == list:
                features = list(X.columns)
                for f in jump_start:
                    features.remove(f)
                
                for feat in features:
                    features2.append(jump_start + [feat])
            features = features2
            
        if jump_start == None:    
            features = []
            for feature in X.columns:
                features.append([feature])
        
        while True:
            start_time = time.time()
            # features as keys, score as values
            feat_com = {}
            print(f'-----------------------------------------------------------Trial {n_trial}-----------------------------------------------------------')
            in_trial_count = 1
            # try out all features
            if n_trial == 1 and jump_start == None:
                for feature in features:
                    if scoring == 'mean_absolute_error':
                        scoring = 'neg_mean_absolute_error'
                    if scoring == 'mean_squared_error':
                        scoring = 'neg_mean_squared_error'    
                    if scoring == 'root_mean_squared_error':
                        scoring = 'neg_root_mean_squared_error'
                    cross_val_score_res = cross_val_score(model, X[feature], y, cv=cv, scoring=scoring)
                    if scoring == 'neg_mean_absolute_error' or scoring == 'neg_mean_squared_error' or 'neg_root_mean_squared_error':
                        score = round(cross_val_score_res.mean(), n_digit) * -1
                    else:
                        score = round(cross_val_score_res.mean(), n_digit)
                    std = round(cross_val_score_res.std(), n_digit)
                    feat_com[feature[0]] = score
                    print(f'{in_trial_count}/{len(features)}: {feature}')
                    if scoring == 'neg_mean_absolute_error':
                        scoring = 'mean_absolute_error'
                    if scoring == 'neg_mean_squared_error':
                        scoring = 'mean_squared_error'    
                    if scoring == 'neg_root_mean_squared_error':
                        scoring = 'root_mean_squared_error'
                    scoring_str = ' '.join(scoring.split('_')).title().replace('Neg', 'Negative').replace('Rand', 'Random').replace('Max', 'Maximum')
                    print(f'      {scoring_str}: {score}, Standard Deviation: {std}')
                    print(' ')
                    in_trial_count += 1
                    
            if n_trial > 1 or jump_start != None:
                for feature in features:
                    if scoring == 'mean_absolute_error':
                        scoring = 'neg_mean_absolute_error'
                    if scoring == 'mean_squared_error':
                        scoring = 'neg_mean_squared_error'    
                    if scoring == 'root_mean_squared_error':
                        scoring = 'neg_root_mean_squared_error'
                    cross_val_score_res = cross_val_score(model, X[feature], y, cv=cv)
                    if scoring == 'neg_mean_absolute_error' or scoring == 'neg_mean_squared_error' or 'neg_root_mean_squared_error':
                        score = round(cross_val_score_res.mean(), n_digit) * -1
                    else:
                        score = round(cross_val_score_res.mean(), n_digit)
                    score = round(cross_val_score_res.mean(), n_digit)
                    std = round(cross_val_score_res.std(), n_digit)
                    feat_com[tuple(feature)] = score
                    print(f'{in_trial_count}/{len(features)}: {feature}')
                    if scoring == 'neg_mean_absolute_error':
                        scoring = 'mean_absolute_error'
                    if scoring == 'neg_mean_squared_error':
                        scoring = 'mean_squared_error'    
                    if scoring == 'neg_root_mean_squared_error':
                        scoring = 'root_mean_squared_error'
                    scoring_str = ' '.join(scoring.split('_')).title().replace('Neg', 'Negative').replace('Rand', 'Random').replace('Max', 'Maximum')
                    print(f'      {scoring_str}: {score}, Standard Deviation: {std}')
                    print(' ')
                    in_trial_count += 1
                    
            # pick the and store trial best
            self.best_com[f'Trial {n_trial}'] = max(feat_com, key=feat_com.get)
            self.best_score[f'Trial {n_trial}'] = max(feat_com.values())

            # define the current trial best
            curr_trial_best = self.best_com[f'Trial {n_trial}']
            
            if n_trial == 1 and jump_start == None:
                # features without the selected trial best
                features.remove([curr_trial_best])
                # generating new combintations of features
                features = [[curr_trial_best]+[i][0] for i in features]
            
            if n_trial > 1 or jump_start != None:
                curr_trial_best2 = list(self.best_com.values())
                features.remove(list(curr_trial_best2[n_trial-1]))
                if type(curr_trial_best2[n_trial-2]) == tuple:
                    
                    # ['PetalWidthCm', ('PetalWidthCm', 'PetalLengthCm'), ('PetalWidthCm', 'PetalLengthCm', 'SepalLengthCm')]
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
            
            # define keys to compare values
            curr_key = max(self.best_score, key=self.best_score.get)
            last_key = f'Trial {n_trial - 1}'
            
            if last_key != 'Trial 0':
                if task == 'classification':
                    if self.best_score[curr_key] < self.best_score[last_key]:
                        chances_to_fail = chances_to_fail - 1
                        
                if task == 'regression':
                    if self.best_score[curr_key] > self.best_score[last_key]:
                        chances_to_fail = chances_to_fail - 1
            
            print(f'Best Combination of Trial {n_trial}: ')
            if type(self.best_com[f'Trial {n_trial}']) == str:
                print('    ',self.best_com[f'Trial {n_trial}'])
                
            if type(self.best_com[f'Trial {n_trial}']) == tuple:
                print('    ',list(self.best_com[f'Trial {n_trial}']))
            print(' ')
            print(f'Best Score of Trial {n_trial}: ')
            print('    ',self.best_score[f'Trial {n_trial}'])
            print(' ')
            self.trial_best[self.best_com[f'Trial {n_trial}']] = self.best_score[f'Trial {n_trial}']
            
            n_trial += 1
            max_round = max_round - 1
            
            end_time = time.time()
            self.trials_time_spend[f'Trial {n_trial - 1}'] = round(end_time - start_time, 2)
            print(f"Time Spent for Trial {n_trial - 1}: {round(end_time - start_time, 2)}(s)")
            print(' ')
                    
            if chances_to_fail <= 0:
                print('Chances to Fail reached. ')
                print('Trial stops.')
                break
            if max_round <= 0:
                print('max_round reached. ')
                print('Trial stops.')
                break
            if len(features) <= 0:
                print('All features combinations have been tried out.')
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
        if task == 'classification':
            self.trial_best = max(self.trial_best, key=self.trial_best.get)
        if task == 'regression':
            self.trial_best = min(self.trial_best, key=self.trial_best.get)
        self.summary = pd.DataFrame([self.best_com, self.best_score, self.trials_time_spend], 
                                    index=['best_com', 'best_score', 'trials_time_spend']).T
        
        # store the result
        print(f'--------------------------------------------------------Trial Summary--------------------------------------------------------')
        try:
            self.res = self.best_com[f'Trial {n_trial}']
            print(f'Best Combination: ')
            print('    ',self.best_com[f'Trial {n_trial}'])
            print(' ')
            print(f'Best Score: ')
            print('    ',self.best_score[f'Trial {n_trial}'])
            print(' ')
        except:
            n_trial = n_trial - 1
            self.res = self.best_com[f'Trial {n_trial}']
            print(f'Best Combination: ')
            print('    ',list(self.best_com[f'Trial {n_trial}']))
            print(' ')
            print(f'Best Score: ')
            print('    ',self.best_score[f'Trial {n_trial}'])
            print(' ')
        print(f'---------------------------------------------End of Recurrsive Features Selection--------------------------------------------')
        
        # visualizing the trials
        sns.set_theme()
        fig, ax = plt.subplots(figsize=(15, 6))  
        sns.lineplot(x=[i + 1 for i in range(len(self.best_com.keys()))], y=self.best_score.values())
        if task == 'classification':
            plt.axvline(x = np.argmax(list(self.best_score.values())) + 1, color='green', linewidth=2, linestyle='--')
        if task == 'regression':
            plt.axvline(x = np.argmin(list(self.best_score.values())) + 1, color='green', linewidth=2, linestyle='--')
        plt.ylabel(f'{scoring_str}')
        plt.xlabel('Trials')
        plt.xticks(range(1,len(self.best_score.values()) + 1))
        sns.despine();
        
        trial_end_time = time.time()
        print(f"Total Time Spent: {round(trial_end_time - trial_start_time, 2)}(s)")
