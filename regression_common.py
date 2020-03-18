import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import os
np.random.seed(39)

SCORE_DOMAINS = ['Global Cognition', 'Language', 'Visuospatial Functioning', 'Memory',
                 'Information Processing Speed', 'Executive Functioning']


def create_score_df(metric, score_type, fold, score):

    df = pd.DataFrame()
    df["Domains"] = SCORE_DOMAINS
    df["Metric"] = metric
    df["Score type"] = score_type
    df["Fold"] = fold
    df["Score"] = score

    return df


def perform_regression(X, y, estimator, my_grid, random_search_cv=False, n_jobs=-1, random_iter=20, pca_fold=False,
                       pca_dir=None, nn=False, callbacks=None, print_r2=False):

    kfold_outer = KFold(n_splits=5)

    scores_df = pd.DataFrame()

    i = 1

    for train, test in kfold_outer.split(X):
        X_train = X[train]
        X_test = X[test]

        kfold_inner = KFold(n_splits=5)

        if pca_fold:

            if not os.path.isfile(pca_dir + 'pc_100_train_cv_' + str(i) + '.npy'):
                pca = PCA(n_components=100)
                X_train = pca.fit_transform(X_train)
                X_test = pca.transform(X_test)
                np.save(pca_dir + 'pc_100_train_cv_' + str(i) + '.npy', X_train)
                np.save(pca_dir + 'pc_100_test_cv_' + str(i) + '.npy', X_test)
                np.save(pca_dir + 'pc_100_train_index_cv_' + str(i) + '.npy', train)
                np.save(pca_dir + 'pc_100_test_index_cv_' + str(i) + '.npy', test)

            else:
                X_train = np.load(pca_dir + 'pc_100_train_cv_' + str(i) + '.npy')
                X_test = np.load(pca_dir + 'pc_100_test_cv_' + str(i) + '.npy')
                train = np.load(pca_dir + 'pc_100_train_index_cv_' + str(i) + '.npy')
                test = np.load(pca_dir + 'pc_100_test_index_cv_' + str(i) + '.npy')

        y_train = y[train]
        y_test = y[test]

        if random_search_cv:
            gs_est = RandomizedSearchCV(estimator=estimator, param_distributions=my_grid, n_jobs=n_jobs,
                                        cv=kfold_inner, random_state=39, n_iter=random_iter)
        else:
            gs_est = GridSearchCV(estimator=estimator, param_grid=my_grid, n_jobs=n_jobs, \
                                  cv=kfold_inner)

        if nn:
            gs_est.fit(X_train, y_train, nn=True, callbacks=callbacks)

        else:
            gs_est.fit(X_train, y_train)

        y_predicted_train = gs_est.predict(X_train)
        y_predicted_test = gs_est.predict(X_test)

        train_rsq_k = r2_score(y_train, y_predicted_train, multioutput='raw_values')
        test_rsq_k = r2_score(y_test, y_predicted_test, multioutput='raw_values')

        train_mae_k = mean_absolute_error(y_train, y_predicted_train, multioutput='raw_values')
        test_mae_k = mean_absolute_error(y_test, y_predicted_test, multioutput='raw_values')

        train_mse_k = mean_squared_error(y_train, y_predicted_train, multioutput='raw_values')
        test_mse_k = mean_squared_error(y_test, y_predicted_test, multioutput='raw_values')

        scores_df = pd.concat([scores_df, create_score_df("R2", "In-sample", i, train_rsq_k)], ignore_index=True)
        scores_df = pd.concat([scores_df, create_score_df("R2", "Out-of-sample", i, test_rsq_k)], ignore_index=True)

        scores_df = pd.concat([scores_df, create_score_df("MAE", "In-sample", i, train_mae_k)], ignore_index=True)
        scores_df = pd.concat([scores_df, create_score_df("MAE", "Out-of-sample", i, test_mae_k)], ignore_index=True)

        scores_df = pd.concat([scores_df, create_score_df("MSE", "In-sample", i, train_mse_k)], ignore_index=True)
        scores_df = pd.concat([scores_df, create_score_df("MSE", "Out-of-sample", i, test_mse_k)], ignore_index=True)

        print('Fold-'+str(i) + ': Best params:', gs_est.best_params_)
        if print_r2:
            print(train_rsq_k)
            print(test_rsq_k)

        i += 1

    return scores_df


def plot_scores(score_df, score_type, metric, hue_order=None):

    plt.figure(figsize=(15, 10))

    data = score_df[(score_df["Score type"] == score_type) & (score_df["Metric"] == metric)]

    # to have consistent colors between analyses
    palette = {"Ridge":"C0", "Multitask Ridge": "C0", "PLS": "C1", "Random Forest": "C2", "CCA": "C3", "Neural Network": "C4",
               "SVR": "C5", "XGBoost": "C6"}

    sns.barplot(x="Domains", y="Score", hue="Model", hue_order=hue_order,
                data=data, ci="sd", capsize=.2, palette=palette)
    plt.axhline(0, color="black")

    if score_type == "Out-of-sample":
        if metric == "R2":
            plt.ylim(-0.2, 0.2)
        elif metric == "MAE":
            plt.ylim(0.0, 1.1)
        elif metric == "MSE":
            plt.ylim(0.0, 1.8)
    elif score_type == "In-sample":
        if metric == "R2":
            plt.ylim(-0.1, 1.1)
        elif metric == "MAE":
            plt.ylim(0.0, 1.0)
        elif metric == "MSE":
            plt.ylim(0.0, 1.4)

    plt.title(metric + " (" + score_type + ")", fontsize=30)
    plt.tight_layout()
    plt.show()


def plot_all_scores(score_df, hue_order=None):
    plot_scores(score_df, "Out-of-sample", "R2", hue_order=hue_order)
    plot_scores(score_df, "Out-of-sample", "MAE", hue_order=hue_order)
    plot_scores(score_df, "Out-of-sample", "MSE", hue_order=hue_order)
    plot_scores(score_df, "In-sample", "R2", hue_order=hue_order)
    plot_scores(score_df, "In-sample", "MAE", hue_order=hue_order)
    plot_scores(score_df, "In-sample", "MSE", hue_order=hue_order)

