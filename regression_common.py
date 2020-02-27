import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display
np.random.seed(39)


SCORE_DOMAINS = ['global cognition', 'language', 'visuospatial functioning', 'memory',\
                 'information processing speed', 'executive functioning']

SCORE_FOLDS = ['Fold-1', 'Fold-2', 'Fold-3', 'Fold-4', 'Fold-5']                 

def perform_regression(X, y, estimator, my_grid, random_search_cv=False):
    kfold_outer = KFold(n_splits=5)
    
    # R^2 
    train_rsq = []
    test_rsq = []
    
    # MAE
    train_mae = []
    test_mae = []
    
    # MSE
    train_mse = []
    test_mse = []
    
    i=1

    for train, test in kfold_outer.split(X):
        X_train = X[train]
        y_train = y[train]
        X_test = X[test]
        y_test = y[test]

        kfold_inner = KFold(n_splits=5)
        
        if random_search_cv:
            gs_est = RandomizedSearchCV(estimator=estimator, param_distributions=my_grid, n_jobs=4, \
                                        cv=kfold_inner, random_state=39, n_iter=20)
        else:
            gs_est = GridSearchCV(estimator=estimator, param_grid=my_grid, n_jobs=4, cv=kfold_inner)
        
        gs_est.fit(X_train, y_train)

        y_predicted_train = gs_est.predict(X_train)
        y_predicted_test = gs_est.predict(X_test)
        
        train_rsq_k = r2_score(y_train, y_predicted_train, multioutput='raw_values')
        test_rsq_k = r2_score(y_test, y_predicted_test, multioutput='raw_values')
        
        train_rsq.append(train_rsq_k)
        test_rsq.append(test_rsq_k)
        
        train_mae_k = mean_absolute_error(y_train, y_predicted_train, multioutput='raw_values')
        test_mae_k = mean_absolute_error(y_test, y_predicted_test, multioutput='raw_values')
        
        train_mae.append(train_mae_k)
        test_mae.append(test_mae_k)
        
        train_mse_k = mean_squared_error(y_train, y_predicted_train, multioutput='raw_values')
        test_mse_k = mean_squared_error(y_test, y_predicted_test, multioutput='raw_values')
        
        train_mse.append(train_mse_k)
        test_mse.append(test_mse_k)
        
        print('\nFold-'+str(i) + ': Best params:', gs_est.best_params_)     
        i+=1
                 
    train_rsq_df = pd.DataFrame(train_rsq, SCORE_FOLDS, SCORE_DOMAINS).transpose()
    train_rsq_df['mean'] = train_rsq_df.mean(axis=1)
    
    test_rsq_df = pd.DataFrame(test_rsq, SCORE_FOLDS, SCORE_DOMAINS).transpose()
    test_rsq_df['mean'] = test_rsq_df.mean(axis=1)
    
    train_mae_df = pd.DataFrame(train_mae, SCORE_FOLDS, SCORE_DOMAINS).transpose()
    train_mae_df['mean'] = train_mae_df.mean(axis=1)
    
    test_mae_df = pd.DataFrame(test_mae, SCORE_FOLDS, SCORE_DOMAINS).transpose()
    test_mae_df['mean'] = test_mae_df.mean(axis=1)
    
    train_mse_df = pd.DataFrame(train_mse, SCORE_FOLDS, SCORE_DOMAINS).transpose()
    train_mse_df['mean'] = train_mse_df.mean(axis=1)
    
    test_mse_df = pd.DataFrame(test_mse, SCORE_FOLDS, SCORE_DOMAINS).transpose()
    test_mse_df['mean'] = test_mse_df.mean(axis=1)
    
    return ((train_rsq_df, test_rsq_df), (train_mae_df, test_mae_df), (train_mse_df, test_mse_df))

def display_results(score_train, score_test, metric, plot_only=True):

    if not plot_only:

        print('\n' + metric + ' (in-sample):')
        display(score_train)
        print('\n' + metric + ' (out-of-sample):')
        display(score_test)
    
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    
    sns.barplot(data=score_train.transpose().drop(['mean'], axis=0), ci="sd", capsize=.2, ax=ax[0])
    ax[0].set_xticklabels(ax[0].get_xticklabels(), rotation=45, ha="right")
    ax[0].set_title(metric + ' (in-sample):')
    ax[0].set_ylabel(metric)
    
    sns.barplot(data=score_test.transpose().drop(['mean'], axis=0), ci="sd", capsize=.2, ax=ax[1])
    ax[1].set_xticklabels(ax[1].get_xticklabels(), rotation=45, ha="right")
    ax[1].set_title(metric + ' (out-of-sample):')
    ax[1].set_ylabel(metric)
    
    fig.tight_layout()

def display_all_results(scores, plot_only=True):

    display_results(scores[0][0], scores[0][1], "R^2", plot_only)
    display_results(scores[1][0], scores[1][1], "MAE", plot_only)
    display_results(scores[2][0], scores[2][1], "MSE", plot_only)
