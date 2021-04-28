#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import math

from sklearn.model_selection import cross_val_score

import itertools
# itertools
# get_dict
# combination
import reprlib
r = reprlib.Repr()
r.maxstring = 100

def get_dict(keys,vals):
    return {keys[i]: vals[i] for i in range(len(keys))}

def combination(feat_track_input, score_track_input, test_list_input): # [(a,b), (a,c)] -> [[a,b], [a,c]
    feat_track2_input = []
    for i in feat_track_input:
        feat_track2_input.append(('-').join(i))

    temp = max(get_dict(feat_track2_input,score_track_input).values()) # find max in dict
    res = [key for key in get_dict(feat_track2_input,score_track_input) if get_dict(feat_track2_input,score_track_input)[key] == temp] # find max in dict

    best_com = res[0].split('-')
    test_list2_input = test_list_input.copy()
    for i in range(0, len(best_com)):
        test_list2_input.remove(best_com[i])

    best_com2 = [i for i in itertools.repeat(best_com, len(test_list2_input))]

    return (np.hstack([pd.DataFrame(best_com2),pd.DataFrame(test_list2_input)])).tolist()

class RecurrsiveFeatureSelector:
    def __init__(self):
        pass
        
    def search(self, model, X, y, cv, task, scoring, k = None, n = 0, jump_start = False, base_com = None):
        if task == 'classification':
            if jump_start == True and base_com == None:
                print('Please input base_com as list.')

            if jump_start == False: 
                feat_track, score_track, count, no_com = [], [], 1, X.shape[1]  

                for i in X.columns:
                    score = cross_val_score(model, X[[i]], y.values.ravel(), cv=cv)
                    feat_track.append(i)
                    score_track.append(score.mean())
                    print(count, str('/'),no_com, str(' '), i ,str(': '))
                    print('         Score = ', score.mean(), str(',  '), str('Standard Deviation = '), round(score.std(ddof=0), 3))
                    print(' ')
                    count = count + 1

                score_dict = get_dict(feat_track,score_track)

                performance = pd.DataFrame([score_dict],index=['score']).T.sort_values('score',ascending=False)

                test_list = list(performance.index)

                best_feat = test_list[0]

                print(' ')
                print(f'Best feature to start with: {best_feat}')
                print('(According to scoring)')
                print(' ')
            #-------------------------------------------------------Trial 1--------------------------------------------------------------

                score_track, feat_track, best_score_track, best_combinations_track, no_trial, model = [], [], [], [], 1, model

                print(f'RFS of {model} started:')
                print(' ')
                print(f'■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Trial {no_trial} ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
                count = 1


                for i in itertools.product(test_list[0:1],test_list[1:]):
                    score = cross_val_score(model, X[(list(i))], y.values.ravel(), cv=cv)
                    feat_track.append(list(i))
                    score_track.append(score.mean())
                    print(count, str('/'),len(test_list[1:]), str(' ['), r.repr('", "'.join(list(i))), str(']'))
                    print('          Score = ', score.mean(), str(',  '), str('Standard Deviation = '), round(score.std(ddof=0), 3))
                    print(' ')
                    count = count + 1

                print(' ')
                print(f'Best of Trial {no_trial}:')
                print(feat_track[np.argmax(score_track)])
                print('Best Score: ', np.array(score_track).max())
                print(' ')

                best_combinations_track.append(feat_track[np.argmax(score_track)])
                best_score_track.append(np.array(score_track).max())

            #-------------------------------------------------Trial 1+-------------------------------------------------------------------
            no_trial = 2
            running =True
            #---------------------------------------------Jump Start---------------------------------------------------------------
            if jump_start == True:
                no_trial = 1
                score = cross_val_score(model, X[base_com], y.values.ravel(), cv=cv)
                score_track2, feat_track2, count, best_combinations_track, best_score_track = [], [], 1, [], []

                best_combinations_track.append(base_com)
                best_score_track.append(score.mean())

                print(' ')
                print('Pre-Trial Combination:')
                print(' ')
                print(best_combinations_track[0])
                print(' ')
                print('Score: ', score.mean())
                print(' ')


                print(f'■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Trial {no_trial} ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
                print(' ')

                df1 = pd.DataFrame([i for i in itertools.repeat(base_com, len(X.drop(base_com,axis=1).columns[1:]))])
                df2 = pd.DataFrame(X.drop(base_com,axis=1).columns[1:])

                for i in np.hstack([df1,df2]).tolist():
                        score = cross_val_score(model, X[i], y.values.ravel(), cv=cv)
                        feat_track2.append(i)
                        score_track2.append(score.mean())
                        print(count, str('/'),len(np.hstack([df1,df2]).tolist()), str(' ['), r.repr('", "'.join(i)), str(']'))
                        print('          Score = ', score.mean(), str(',  '), str('Standard Deviation: '), round(score.std(ddof=0), 3))
                        print(' ')
                        count = count + 1

                print(' ')
                print(f'Best of Trial {no_trial}:')
                print(feat_track2[np.argmax(score_track2)])
                print('Best Score: ', np.array(score_track2).max())

                best_combinations_track.append(feat_track2[np.argmax(score_track2)])
                best_score_track.append(np.array(score_track2).max())
                no_trial = no_trial + 1
                test_list = list(X.columns)

                if len(combination(feat_track2, score_track2, test_list)) == 1 or best_score_track[no_trial - 2] > best_score_track[no_trial - 1]:
                    if n == 0:
                        # Trial Summary
                        fcom = best_combinations_track[np.argmax(best_score_track)]
                        fscore = np.array(best_score_track).max()
                        print(' ')
                        print(f'Best overall combination: {fcom}')
                        print(' ')
                        print(f'Best overall score: {fscore}')
                        sns.lineplot(x=np.arange(0, no_trial, 1), y=best_score_track)
                        plt.axvline(x = np.argmax(best_score_track), color='green', linewidth=2, linestyle='--')
                        plt.xlabel('Number of Trials')
                        plt.ylabel('Score')
                        sns.despine();
                        print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ End of RFS ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')

            #---------------------------------------------------------------------------------------------------------------------
            if k == None:
                k = np.inf

            if jump_start ==True:
                feat_track = feat_track2
                score_track = score_track2

            while running == True and len(combination(feat_track, score_track, test_list)) < k + 1:

                score_track2, feat_track2, count = [], [], 1

                print(' ')

                if len(feat_track[0]) == len(test_list):
                    break

                print(f'■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Trial {no_trial} ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')

                for i in combination(feat_track, score_track, test_list):
                    score = cross_val_score(model, X[i], y.values.ravel(), cv=cv)
                    feat_track2.append(i)
                    score_track2.append(score.mean())
                    print(count, str('/'),len(combination(feat_track, score_track, test_list)), str(' ['), r.repr('", "'.join(i)), str(']'))
                    print('          Score = ', score.mean(), str(',  '), str('Standard Deviation: '), round(score.std(ddof=0), 3))
                    print(' ')
                    count = count + 1

                print(' ')
                print(f'Best of Trial {no_trial}:')
                print(feat_track2[np.argmax(score_track2)])
                print('Best Score: ', np.array(score_track2).max())

                best_combinations_track.append(feat_track2[np.argmax(score_track2)])
                best_score_track.append(np.array(score_track2).max())

                feat_track = feat_track2

                score_track = [i for i in score_track2]

                if best_score_track[no_trial - 2] >= best_score_track[no_trial - 1]:
                    n = n - 1

                if len(feat_track[0]) == len(test_list):
                    break

                if (len(combination(feat_track, score_track, test_list))==1 and best_score_track[no_trial - 2] > best_score_track[no_trial - 1]) or best_score_track[no_trial - 2] > best_score_track[no_trial - 1] or len(feat_track) == 1:
                    no_trial = no_trial + 1
                    break

                no_trial = no_trial + 1
            #-------------------------------------------------Trial n+-------------------------------------------------------------------
                while n > 0:

                    score_track2, feat_track2 = [], []

                    print(' ')                                                                                 
                    print(f'■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Trial {no_trial} (Extra Trial) ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
                    count = 1

                    for i in combination(feat_track, score_track, test_list):
                        score = cross_val_score(model, X[i], y.values.ravel(), cv=cv)
                        feat_track2.append(i)
                        score_track2.append(score.mean())
                        print(count, str('/'),len(combination(feat_track, score_track, test_list)), str(' ['), r.repr('", "'.join(i)), str(']'))
                        print('          Score = ', score.mean(), str(',  '), str('Standard Deviation: '), round(score.std(ddof=0), 3))
                        print(' ')
                        count = count + 1                    

                    print(' ')
                    print(f'Best of Trial {no_trial} (Extra Trial):')
                    print(feat_track2[np.argmax(score_track2)])
                    print('Best Score: ', np.array(score_track2).max())

                    best_combinations_track.append(feat_track2[np.argmax(score_track2)])
                    best_score_track.append(np.array(score_track2).max())

                    feat_track = feat_track2

                    score_track = [i for i in score_track2]

                    n = n - 1

                    if best_score_track[no_trial - 2] > best_score_track[no_trial - 1]:
                        n = n + 1

                    if len(feat_track[0]) == len(test_list) or n==0:
                        break

                    no_trial = no_trial + 1    


            # Trial Summary
            fcom = best_combinations_track[np.argmax(best_score_track)]
            fscore = np.array(best_score_track).max()
            print(' ')
            print(f'Best overall combination: {fcom}')
            print(' ')
            print(f'Best overall score: {fscore}')        
            print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ End of RFS ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
            self.best_combination = fcom
            self.best_score = fscore
            self.best_score_track = best_score_track
            self.no_trial = no_trial
    
        if task == 'regression':
            if jump_start == True and base_com == None:
                print('Please input base_com as list.')

            if jump_start == False: 
                feat_track, score_track, count, no_com = [], [], 1, X.shape[1]  

                for i in X.columns:
                    score = cross_val_score(model, X[[i]], y.values.ravel(), cv=cv, scoring='neg_mean_absolute_error')
                    feat_track.append(i)
                    score_track.append(score.mean())
                    print(count, str('/'),no_com, str(' '), i ,str(': '))
                    print('         Score = ', score.mean(), str(',  '), str('Standard Deviation = '), round(score.std(ddof=0), 3))
                    print(' ')
                    count = count + 1

                score_dict = get_dict(feat_track,score_track)

                performance = pd.DataFrame([score_dict],index=['score']).T.sort_values('score',ascending=True)

                test_list = list(performance.index)

                best_feat = test_list[0]

                print(' ')
                print(f'Best feature to start with: {best_feat}')
                print('(According to scoring)')
                print(' ')
            #-------------------------------------------------------Trial 1--------------------------------------------------------------

                score_track, feat_track, best_score_track, best_combinations_track, no_trial, model = [], [], [], [], 1, model

                print(f'RFS of {model} started:')
                print(' ')
                print(f'■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Trial {no_trial} ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
                count = 1


                for i in itertools.product(test_list[0:1],test_list[1:]):
                    score = cross_val_score(model, X[(list(i))], y.values.ravel(), cv=cv, scoring='neg_mean_absolute_error')
                    feat_track.append(list(i))
                    score_track.append(score.mean())
                    print(count, str('/'),len(test_list[1:]), str(' ['), r.repr('", "'.join(list(i))), str(']'))
                    print('          Score = ', score.mean(), str(',  '), str('Standard Deviation = '), round(score.std(ddof=0), 3))
                    print(' ')
                    count = count + 1

                print(' ')
                print(f'Best of Trial {no_trial}:')
                print(feat_track[np.argmin(score_track)])
                print('Best Score: ', np.array(score_track).min())
                print(' ')

                best_combinations_track.append(feat_track[np.argmin(score_track)])
                best_score_track.append(np.array(score_track).min())

            #-------------------------------------------------Trial 1+-------------------------------------------------------------------
            no_trial = 2
            running =True
            #---------------------------------------------Jump Start---------------------------------------------------------------
            if jump_start == True:
                no_trial = 1
                score = cross_val_score(model, X[base_com], y.values.ravel(), cv=3, scoring='neg_mean_absolute_error')
                score_track2, feat_track2, count, best_combinations_track, best_score_track = [], [], 1, [], []

                best_combinations_track.append(base_com)
                best_score_track.append(score.mean())

                print(' ')
                print('Pre-Trial Combination:')
                print(' ')
                print(best_combinations_track[0])
                print(' ')
                print('Score: ', score.mean())
                print(' ')


                print(f'■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Trial {no_trial} ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
                print(' ')

                df1 = pd.DataFrame([i for i in itertools.repeat(base_com, len(X.drop(base_com,axis=1).columns[1:]))])
                df2 = pd.DataFrame(X.drop(base_com,axis=1).columns[1:])

                for i in np.hstack([df1,df2]).tolist():
                        score = cross_val_score(model, X[i], y.values.ravel(), cv=cv, scoring='neg_mean_absolute_error')
                        feat_track2.append(i)
                        score_track2.append(score.mean())
                        print(count, str('/'),len(np.hstack([df1,df2]).tolist()), str(' ['), r.repr('", "'.join(i)), str(']'))
                        print('          Score = ', score.mean(), str(',  '), str('Standard Deviation: '), round(score.std(ddof=0), 3))
                        print(' ')
                        count = count + 1

                print(' ')
                print(f'Best of Trial {no_trial}:')
                print(feat_track2[np.argmin(score_track2)])
                print('Best Score: ', np.array(score_track2).min())

                best_combinations_track.append(feat_track2[np.argmin(score_track2)])
                best_score_track.append(np.array(score_track2).min())
                no_trial = no_trial + 1
                test_list = list(X.columns)

                if len(combination(feat_track2, score_track2, test_list)) == 1 or best_score_track[no_trial - 2] > best_score_track[no_trial - 1]:
                    if n == 0:
                        # Trial Summary
                        fcom = best_combinations_track[np.argmin(best_score_track)]
                        fscore = np.array(best_score_track).min()
                        print(' ')
                        print(f'Best overall combination: {fcom}')
                        print(' ')
                        print(f'Best overall score: {fscore}')
                        sns.lineplot(x=np.arange(0, no_trial, 1), y=best_score_track)
                        plt.axvline(x = np.argmin(best_score_track), color='green', linewidth=2, linestyle='--')
                        plt.xlabel('Number of Trials')
                        plt.ylabel('Score')
                        sns.despine();
                        print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ End of RFS ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')

            #---------------------------------------------------------------------------------------------------------------------
            if k == None:
                k = np.inf

            if jump_start ==True:
                feat_track = feat_track2
                score_track = score_track2

            while running == True and len(combination(feat_track, score_track, test_list)) < k + 1:

                score_track2, feat_track2, count = [], [], 1

                print(' ')

                if len(feat_track[0]) == len(test_list):
                    break

                print(f'■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Trial {no_trial} ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')

                for i in combination(feat_track, score_track, test_list):
                    score = cross_val_score(model, X[i], y.values.ravel(), cv=cv, scoring='neg_mean_absolute_error')
                    feat_track2.append(i)
                    score_track2.append(score.mean())
                    print(count, str('/'),len(combination(feat_track, score_track, test_list)), str(' ['), r.repr('", "'.join(i)), str(']'))
                    print('          Score = ', score.mean(), str(',  '), str('Standard Deviation: '), round(score.std(ddof=0), 3))
                    print(' ')
                    count = count + 1

                print(' ')
                print(f'Best of Trial {no_trial}:')
                print(feat_track2[np.argmin(score_track2)])
                print('Best Score: ', np.array(score_track2).min())

                best_combinations_track.append(feat_track2[np.argmin(score_track2)])
                best_score_track.append(np.array(score_track2).min())

                feat_track = feat_track2

                score_track = [i for i in score_track2]

                if best_score_track[no_trial - 2] >= best_score_track[no_trial - 1]:
                    n = n - 1

                if len(feat_track[0]) == len(test_list):
                    break

                if (len(combination(feat_track, score_track, test_list))==1 and best_score_track[no_trial - 2] > best_score_track[no_trial - 1]) or best_score_track[no_trial - 2] > best_score_track[no_trial - 1] or len(feat_track) == 1:
                    no_trial = no_trial + 1
                    break

                no_trial = no_trial + 1
            #-------------------------------------------------Trial n+-------------------------------------------------------------------
                while n > 0:

                    score_track2, feat_track2 = [], []

                    print(' ')                                                                                 
                    print(f'■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ Trial {no_trial} (Extra Trial) ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
                    count = 1

                    for i in combination(feat_track, score_track, test_list):
                        score = cross_val_score(model, X[i], y.values.ravel(), cv=cv, scoring='neg_mean_absolute_error')
                        feat_track2.append(i)
                        score_track2.append(score.mean())
                        print(count, str('/'),len(combination(feat_track, score_track, test_list)), str(' ['), r.repr('", "'.join(i)), str(']'))
                        print('          Score = ', score.mean(), str(',  '), str('Standard Deviation: '), round(score.std(ddof=0), 3))
                        print(' ')
                        count = count + 1                    

                    print(' ')
                    print(f'Best of Trial {no_trial} (Extra Trial):')
                    print(feat_track2[np.argmin(score_track2)])
                    print('Best Score: ', np.array(score_track2).min())

                    best_combinations_track.append(feat_track2[np.argmin(score_track2)])
                    best_score_track.append(np.array(score_track2).min())

                    feat_track = feat_track2

                    score_track = [i for i in score_track2]

                    n = n - 1

                    if best_score_track[no_trial - 2] > best_score_track[no_trial - 1]:
                        n = n + 1

                    if len(feat_track[0]) == len(test_list) or n==0:
                        break

                    no_trial = no_trial + 1    


            # Trial Summary
            fcom = best_combinations_track[np.argmin(best_score_track)]
            fscore = np.array(best_score_track).min()
            print(' ')
            print(f'Best overall combination: {fcom}')
            print(' ')
            print(f'Best overall score: {fscore}')        
            print('■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■ End of RFS ■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■■')
            self.best_combination = fcom
            self.best_score = fscore
            self.best_score_track = best_score_track
            self.no_trial = no_trial
    
        if task == 'classification':
            fig, ax = plt.subplots(figsize=(15, 6))
            sns.lineplot(x=np.arange(1, len(self.best_score_track) + 1), y=self.best_score_track)
            plt.axvline(x = np.argmax(self.best_score_track) + 1, color='green', linewidth=2, linestyle='--')
            plt.xlabel('Number of Trials')
            plt.ylabel('Score')
            plt.xticks(range(1,len(self.best_score_track) + 1))
            sns.despine();
            
        if task == 'regression':
            fig, ax = plt.subplots(figsize=(15, 6))
            sns.lineplot(x=np.arange(1, len(self.best_score_track) + 1), y=self.best_score_track)
            plt.axvline(x = np.argmin(self.best_score_track) + 1, color='green', linewidth=2, linestyle='--')
            plt.xlabel('Number of Trials')
            plt.ylabel('Score')
            plt.xticks(range(1,len(self.best_score_track) + 1))
            sns.despine();
