import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score
import math


import seaborn as sns
np.random.seed(39)

SCORE_DOMAINS = ['Global Cognition', 'Language', 'Visuospatial Functioning', 'Memory',
                 'Information Processing Speed', 'Executive Functioning']


def fit_classifier(X, y, estimator, my_grid, model_name, percentile, i_domain, random_search_cv=False, n_jobs=-1, random_iter=20,
                   nn=False, callbacks=None):

    kfold_outer = KFold(n_splits=5, shuffle=True, random_state=39)

    scores = []
    best_params = []

    i_fold = 1

    for train, test in kfold_outer.split(X):
        X_train = X[train]
        X_test = X[test]

        kfold_inner = KFold(n_splits=5)

        y_train = y[train]
        y_test = y[test]

        if random_search_cv:
            gs_est = RandomizedSearchCV(estimator=estimator, param_distributions=my_grid, n_jobs=n_jobs,
                                        cv=kfold_inner, random_state=39, n_iter=random_iter)
        else:
            gs_est = GridSearchCV(estimator=estimator, param_grid=my_grid, n_jobs=n_jobs,
                                  cv=kfold_inner)

        if nn:
            gs_est.fit(X_train, y_train, nn=True, callbacks=callbacks)

        else:
            gs_est.fit(X_train, y_train)

        y_predicted_train = gs_est.predict(X_train)
        y_predicted_test = gs_est.predict(X_test)

        train_acc_k = accuracy_score(y_train, y_predicted_train)
        test_acc_k = accuracy_score(y_test, y_predicted_test)

        train_auc_k = roc_auc_score(y_train, y_predicted_train)
        test_auc_k = roc_auc_score(y_test, y_predicted_test)

        scores.append(["Accuracy", "In-sample", i_fold, train_acc_k])
        scores.append(["Accuracy", "Out-of-sample", i_fold, test_acc_k])
        scores.append(["AUC", "In-sample", i_fold, train_auc_k])
        scores.append(["AUC", "Out-of-sample", i_fold, test_auc_k])

        best_params.append([i_fold, gs_est.best_params_])

        print('Fold-'+str(i_fold) + ': Best params:', gs_est.best_params_)

        i_fold += 1

    scores_df = pd.DataFrame(scores, columns=["Metric", "Score type", "Fold", "Score"])
    scores_df.insert(0, "Domain", SCORE_DOMAINS[i_domain])
    scores_df.insert(0, "% Data", percentile)
    scores_df.insert(0, "Model", model_name)

    best_params_df = pd.DataFrame(best_params, columns=["Fold", "Best params"])
    best_params_df.insert(0, "Domain", SCORE_DOMAINS[i_domain])
    best_params_df.insert(0, "% Data", percentile)
    best_params_df.insert(0, "Model", model_name)

    return scores_df, best_params_df, gs_est


def run_classification(X, Y, Y_sort_idx, estimator, grid, model_name, random_search_cv=False):

    scores_df_all = pd.DataFrame()
    best_params_df_all = pd.DataFrame()

    percentiles = [.50, .30, .20, .10]

    for i_domain in range(Y.shape[1]):

        y = Y[:, i_domain]

        print('\n' + SCORE_DOMAINS[i_domain] + '\n')

        for p in percentiles:

            X_sorted = X[Y_sort_idx[:, i_domain]]

            X_low = X_sorted[0: math.floor(X_sorted.shape[0] * p), :]
            X_high = X_sorted[math.ceil(X_sorted.shape[0] * (1 - p)):, :]

            X_new = np.append(X_low, X_high, axis=0)
            y_new = np.append(np.zeros(len(X_low)), np.ones(len(X_high)), axis=0)

            print("Data % used: " + str(p*2*100) + "%")

            scores_df, best_param_df, est = fit_classifier(X_new, y_new, estimator, grid, model_name, p*2*100, i_domain,
                                                           random_search_cv=random_search_cv)

            scores_df_all = pd.concat([scores_df_all, scores_df], ignore_index=True)
            best_params_df_all = pd.concat([best_params_df_all, best_param_df], ignore_index=True)

    return scores_df_all, best_params_df_all


def plot_scores(score_df, score_type, hue_order=None):

    data = score_df[(score_df["Score type"] == score_type)]

    palette = {"Logistic Regression": "C0", "Random Forest": "C2", "Neural Network": "C4", "XGBoost": "C3"}

    g = sns.catplot(x="Domain", y="Score", hue="Model", col="% Data", row="Metric",
                    data=data, ci="sd", kind="bar", palette=palette, hue_order=hue_order)

    g.set_axis_labels("", "Score (Mean and Standard Error across 5 CV folds")

    for i, ax in enumerate(g.fig.axes):
        ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
        ax.axhline(0.5, color="black")

    g.fig.suptitle("Classification results: " + score_type, y=1.08, fontsize=30)
    g.savefig('sdas')


def plot_all_scores(score_df, hue_order=None):
    plot_scores(score_df, "Out-of-sample", hue_order=hue_order)
    plot_scores(score_df, "In-sample", hue_order=hue_order)
