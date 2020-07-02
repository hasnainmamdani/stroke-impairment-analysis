import math
import os
import random

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# To ensure reproducibility
random.seed(39)
np.random.seed(39)


def get_PC_data(data_dir):
    # Calculate 100 PC components
    if not os.path.isfile(data_dir + 'binary_imgs_pc_100.npy'):
        imgs = get_binary_imgs(data_dir)
        print("Created/loaded binary images. Size:", imgs.shape)

        pca = PCA(n_components=100, copy=False)
        X_pc = pca.fit_transform(imgs)
        np.save(data_dir + 'binary_imgs_pc_100.npy', X_pc)

    else:
        X_pc = np.load(data_dir + 'binary_imgs_pc_100.npy')
        print("Created/loaded 100 PC components. Size:", X_pc.shape)

    return X_pc


def get_binary_imgs(data_dir):
    # Create/store/load binary image data
    if not os.path.isfile(data_dir + 'binary_imgs.npy'):
        from nilearn.input_data import NiftiMasker
        from nilearn.datasets import load_mni152_brain_mask
        import glob

        dataset_path = data_dir + "HallymBundang_lesionmaps_Bzdok_n1401/"
        img_filenames = glob.glob(os.path.join(dataset_path, '*.nii.gz'))
        img_filenames.sort()
        print('Number of subjects: %d' % len(img_filenames))

        mask_img = load_mni152_brain_mask()
        masker = NiftiMasker(mask_img=mask_img, memory='nilearn_cache', verbose=5)
        masker = masker.fit()

        imgs = masker.transform(img_filenames)  # break down into slices if necessary
        imgs = imgs.astype(bool)
        np.save(data_dir + 'binary_imgs', imgs)

    else:
        imgs = np.load(data_dir + 'binary_imgs.npy')

    return imgs


def get_patient_scores(data_dir):
    patient_df = pd.read_hdf(data_dir + 'patients.h5', 'p')
    scaler_Y = StandardScaler()
    return scaler_Y.fit_transform(np.array(patient_df[patient_df.columns[5:11]]))


# common library
def mixup_data(X, Y, alpha=0.1, mul_factor=10):

    rs = np.random.RandomState(39)
    n = X.shape[0]

    if len(Y.shape) == 1:
        Y = Y[:, None]

    mixed_X = np.empty((n*(mul_factor-1), X.shape[1]))
    mixed_Y = np.empty((n*(mul_factor-1), Y.shape[1]))

    for i in range(mul_factor-1):

        # sample more than needed as some will be filtered out
        lam = np.random.beta(alpha, alpha, size=round(n*2))

        # original data vectors will be concatenated later
        lam = lam[(lam != 0.0) & (lam != 1.0)][:n][:, None]  # shape nx1

        shuffle_idx = rs.choice(np.arange(n), n, replace=False)

        mixed_X[i*n: (i+1)*n] = lam * X + (1 - lam) * X[shuffle_idx]
        mixed_Y[i*n: (i+1)*n] = lam * Y + (1 - lam) * Y[shuffle_idx]

    # concatenate original data vectors
    mixed_X = np.append(mixed_X, X, axis=0)
    mixed_Y = np.append(mixed_Y, Y, axis=0)

    if mixed_Y.shape[1] == 1:
        mixed_Y = np.ravel(mixed_Y)

    return mixed_X, mixed_Y


def mixup_within_class(X, y, alpha=0.1, mul_factor=10):

    # by mixing-up samples of the same class, the label remains unchanged.

    y_zeros_idx = np.where(y == 0)[0]
    y_ones_idx = np.where(y == 1)[0]

    mixed_X_low, mixed_y_low = mixup_data(X[y_zeros_idx], y[y_zeros_idx], alpha=alpha, mul_factor=mul_factor)
    mixed_X_high, mixed_y_high = mixup_data(X[y_ones_idx], y[y_ones_idx], alpha=alpha, mul_factor=mul_factor)

    return np.concatenate((mixed_X_low, mixed_X_high)), np.concatenate((mixed_y_low, mixed_y_high))


SCORE_DOMAINS = ['Global Cognition', 'Language', 'Visuospatial Functioning', 'Memory',
                 'Information Processing Speed', 'Executive Functioning']


# get top and bottom p% of the data based on target variable and truncate the date in between
def get_extremes_sorted(X, y, p):

    assert len(y.shape) == 1, 'y must be one dimensional'

    sort_idx = np.argsort(y)

    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    extremes_idx = list(range(0, math.floor(X_sorted.shape[0] * p))) + \
                   list(range(math.ceil(X_sorted.shape[0] * (1 - p)), len(sort_idx)))

    return X_sorted[extremes_idx], y_sorted[extremes_idx]


def calculate_classification_metrics(est_cv, X_train, X_test, y_train, y_test, model_name, i_fold, score_domain=None,
                                 score_insample_orig=False, X_train_orig=None, y_train_orig=None):

    scores = []

    y_predicted_train = est_cv.predict(X_train)
    y_predicted_test = est_cv.predict(X_test)

    train_acc = accuracy_score(y_train, y_predicted_train)
    test_acc = accuracy_score(y_test, y_predicted_test)

    y_predicted_prob_train = est_cv.predict_proba(X_train)
    y_predicted_prob_test = est_cv.predict_proba(X_test)

    train_auc = roc_auc_score(y_train, y_predicted_prob_train[:, 1])
    test_auc = roc_auc_score(y_test, y_predicted_prob_test[:, 1])

    if score_insample_orig:
        y_predicted_train_orig = est_cv.predict(X_train_orig)
        train_orig_acc = accuracy_score(y_train_orig, y_predicted_train_orig)

        y_predicted_prob_train_orig = est_cv.predict_proba(X_train_orig)
        train_orig_auc = roc_auc_score(y_train_orig, y_predicted_prob_train_orig[:, 1])

    scores.append([model_name, score_domain, "Accuracy", "In-sample", i_fold, train_acc])
    scores.append([model_name, score_domain, "Accuracy", "Out-of-sample", i_fold, test_acc])
    scores.append([model_name, score_domain, "AUC", "In-sample", i_fold, train_auc])
    scores.append([model_name, score_domain, "AUC", "Out-of-sample", i_fold, test_auc])
    if score_insample_orig:
        scores.append([model_name, score_domain, "Accuracy", "In-sample (original)", i_fold, train_orig_acc])
        scores.append([model_name, score_domain, "AUC", "In-sample (original)", i_fold, train_orig_auc])

    return scores


def run_classification_single_output(X, Y, estimator, grid, model_name, random_search_cv=False, n_jobs=-1,
                                 random_iter=20, nn=False, callbacks=None, print_best_params=True,
                                 mixup=False, mixup_alpha=0.1, mixup_mul_factor=5, use_extreme_all_perc=0.5,
                                 log_X=False, zscore_X=False, score_insample_orig=True):

    scores = []
    best_params = []

    for i_domain in range(Y.shape[1]):

        print(SCORE_DOMAINS[i_domain])

        X_domain = np.copy(X)
        y_domain = np.copy(Y[:, i_domain])

        X_tmp, _ = get_extremes_sorted(X_domain, y_domain, use_extreme_all_perc)
        X_domain = X_tmp

        y_domain = np.append(np.zeros(int(len(X_domain)/2)), np.ones(int(len(X_domain)/2)), axis=0)

        # check if first zeroes and then ones?

        kfold_outer = KFold(n_splits=5, shuffle=True, random_state=39)

        i_fold = 1

        # print('\n% Data used - ' + ("100" if use_extreme_train_perc is None else str(use_extreme_train_perc*2*100)) + '\n')

        for train, test in kfold_outer.split(X_domain):

            X_train = X_domain[train]
            X_test = X_domain[test]
            y_train = y_domain[train]
            y_test = y_domain[test]

            X_train_upd = np.copy(X_train)
            y_train_upd = np.copy(y_train)
            X_test_upd = X_test

            if mixup:
                X_train_tmp, y_train_tmp = mixup_within_class(X_train_upd, y_train_upd, alpha=mixup_alpha, mul_factor=mixup_mul_factor)  # two-step to avoid internal bugs
                X_train_upd, y_train_upd = X_train_tmp, y_train_tmp

            if log_X:
                X_train_tmp = np.log(1 + X_train_upd)
                X_train_upd = X_train_tmp
                X_test_tmp = np.log(1 + X_test_upd)
                X_test_upd = X_test_tmp

            if zscore_X:
                scaler = StandardScaler()
                X_train_tmp = scaler.fit_transform(X_train_upd)
                X_train_upd = X_train_tmp
                X_test_tmp = scaler.transform(X_test_upd)
                X_test_upd = X_test_tmp

            kfold_inner = KFold(n_splits=5, shuffle=True, random_state=31)

            if random_search_cv:
                est_cv = RandomizedSearchCV(estimator=estimator, param_distributions=grid, n_jobs=n_jobs,
                                            cv=kfold_inner, random_state=39, n_iter=random_iter)
            else:
                est_cv = GridSearchCV(estimator=estimator, param_grid=grid, n_jobs=n_jobs, cv=kfold_inner)

            print('\nFold-' + str(i_fold) + '\n')

            if nn:
                est_cv.fit(X_train_upd, y_train_upd, nn=True, callbacks=callbacks)

            else:
                est_cv.fit(X_train_upd, y_train_upd)

            if score_insample_orig:
                if log_X:
                    X_train_tmp = np.log(1 + X_train)
                    X_train = X_train_tmp
                if zscore_X:
                    X_train_tmp = scaler.transform(X_train)
                    X_train = X_train_tmp

                scores_fold = calculate_classification_metrics(est_cv, X_train_upd, X_test_upd, y_train_upd, y_test, model_name, i_fold,
                                                           SCORE_DOMAINS[i_domain], score_insample_orig, X_train, y_train)

            else:
                scores_fold = calculate_classification_metrics(est_cv, X_train_upd, X_test_upd, y_train_upd, y_test, model_name, i_fold,
                                                           SCORE_DOMAINS[i_domain])

            scores.extend(scores_fold)

            best_params.append([model_name, SCORE_DOMAINS[i_domain], i_fold, est_cv.best_params_])

            if print_best_params:
                print("Best params:", best_params[-1])

            i_fold += 1

    scores_df = pd.DataFrame(scores, columns=["Model", "Domain", "Metric", "Score type", "Fold", "Score"])
    best_params_df = pd.DataFrame(best_params, columns=["Model", "Domain", "Fold", "Best params"])

    scores_df.insert(0, "alpha", mixup_alpha if mixup else "No mixup")  # or "\u03B1" - alpha symbol
    scores_df.insert(0, "Data mul factor", str(mixup_mul_factor)+"x" if mixup else "No mixup")
    scores_df.insert(0, "% Data", use_extreme_all_perc*2*100)

    best_params_df.insert(0, "alpha", mixup_alpha if mixup else "No mixup")
    best_params_df.insert(0, "Data mul factor", str(mixup_mul_factor)+"x" if mixup else "No mixup")
    best_params_df.insert(0, "% Data", use_extreme_all_perc*2*100)

    return scores_df, best_params_df


def logistic_regression(X, Y, mixup, mixup_alpha=None, mixup_mul_factor=None, use_extreme_all_perc=0.5,
                        log_X=False, zscore_X=False, score_insample_orig=True):

    estimator = LogisticRegression(random_state=39, max_iter=20000)

    if mixup and use_extreme_all_perc == 0.1:
        C = np.logspace(1, 14, 14)

    else:
        C = np.logspace(-3, 10, 14)

    grid = {'C': C}

    return run_classification_single_output(X, Y, estimator, grid, "Logistic Regression", mixup=mixup,
            mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor, use_extreme_all_perc=use_extreme_all_perc,
            log_X=log_X, zscore_X=zscore_X, score_insample_orig=score_insample_orig)


def random_forest(X, Y, mixup, mixup_alpha=None, mixup_mul_factor=None, use_extreme_all_perc=0.5,
                  log_X=False, zscore_X=False, score_insample_orig=True):

    estimator = RandomForestClassifier(random_state = 39)

    n_estimators = [10, 25, 50, 100, 150, 200, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 3500]
    max_features = ['auto', 'sqrt', 'log2']
    max_depth = [2, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, None]
    min_samples_split = [2, 5, 10, 15, 20, 25, 30, 40, 50, 70, 80, 90]
    min_samples_leaf = [1, 2, 4, 8, 12, 15, 18, 21, 24, 27]
    max_samples = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, None]

    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_samples': max_samples}

    return run_classification_single_output(X, Y, estimator, grid, "Random Forest", random_search_cv=True, random_iter=32,
            mixup=mixup, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor, use_extreme_all_perc=use_extreme_all_perc,
            log_X=log_X, zscore_X=zscore_X, score_insample_orig=score_insample_orig)


def plot_scores(score_df, metric, score_type, title="", save_folder="", col_order=None, hue_order=None):

    g = sns.catplot(x="Domain", y="Score", hue="alpha", col="Data mul factor", col_order=col_order,
                    row="% Data", data=score_df, ci="sd", kind="bar", hue_order=hue_order,
                    row_order=[100.0, 80.0, 60.0, 40.0, 20.0])

    g.set_axis_labels("", "Score (Mean and Standard Deviation across 5 CV folds)")

    for i, ax in enumerate(g.fig.axes):
        ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
        ax.axhline(0, color="black")

    g.fig.suptitle(title, y=1.08, fontsize=30)

    if metric == "R2" and score_type == "Out-of-sample":
        # g.set(ylim=(-1, 0.2))
        pass

    if metric == "R2" and (score_type == "In-sample" or score_type == "In-sample (original)"):
        g.set(ylim=(-0.05, 1.0))

    if metric == "MAE":
        g.set(ylim=(0.0, 1.6))

    # plt.show()
    plt.savefig(save_folder + title, bbox_inches="tight")


def plot_all_scores(score_df, title_prefix="", save_folder="", col_order=None, hue_order=None):

    for metric in score_df["Metric"].unique():
        for score_type in score_df["Score type"].unique():
            for model in score_df["Model"].unique():

                filtered_data = score_df[(score_df["Metric"] == metric) & (score_df["Score type"] == score_type)
                                         & (score_df["Model"] == model)]
                title = title_prefix + metric + " - " + score_type + " - " + model
                plot_scores(filtered_data, metric, score_type, title, save_folder, col_order=col_order, hue_order=hue_order)


def run_with_mixup(X, Y, model, mixup_alphas, mixup_mul_factors, without_mixup=True, result_dir="", plot_scores=False,
                   use_extreme_all_perc=None, log_X=False, zscore_X=False, score_insample_orig=True):

    print("use_extreme_all_perc=", use_extreme_all_perc, "log_X=", log_X, "zscore_X=", zscore_X,
          "mixup_alphas=", mixup_alphas, "mixup_mul_factors=", mixup_mul_factors, "without_mixup=", without_mixup,
          "result_dir=", result_dir, "score_insample_orig=", score_insample_orig)

    scores_all = pd.DataFrame()
    best_params_all = pd.DataFrame()

    p_str = "-"+str(use_extreme_all_perc)
    scores_path = result_dir + "classification-singleoutput-pc100-mixup-" + model + p_str + ".h5"
    best_params_path = result_dir + "best-params-classification-singleoutput-pc100-mixup-" + model + p_str + ".h5"

    if without_mixup:
        # also get results without mixup (baseline)
        print("Performing regression without mixup")

        if model == "logit":
            scores_no_mixup, best_params_no_mixup = logistic_regression(X, Y, mixup=False,
                                                    use_extreme_all_perc=use_extreme_all_perc, log_X=log_X,
                                                    zscore_X=zscore_X, score_insample_orig=score_insample_orig)

        elif model == "rf":
            scores_no_mixup, best_params_no_mixup = random_forest(X, Y, mixup=False,
                                                    use_extreme_all_perc=use_extreme_all_perc, log_X=log_X,
                                                    zscore_X=zscore_X, score_insample_orig=score_insample_orig)

        else:
            print("Error: unrecognizable model")
            raise SystemExit

        scores_all = pd.concat([scores_all, scores_no_mixup], ignore_index=True)
        best_params_all = pd.concat([best_params_all, best_params_no_mixup], ignore_index=True)

        scores_all.to_hdf(scores_path, key='p', mode='w')
        best_params_all.to_hdf(best_params_path, key='p', mode='w')

    for mixup_mul_factor in mixup_mul_factors:

        for mixup_alpha in mixup_alphas:

            print("\n\nData mul factor:", str(mixup_mul_factor)+"x,", "Mixup alpha:", mixup_alpha)

            if model == "logit":
                scores, best_params = logistic_regression(X, Y, mixup=True, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                                            use_extreme_all_perc=use_extreme_all_perc, log_X=log_X, zscore_X=zscore_X,
                                            score_insample_orig=score_insample_orig)

            elif model == "rf":
                scores, best_params = random_forest(X, Y, mixup=True, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                                            use_extreme_all_perc=use_extreme_all_perc, log_X=log_X, zscore_X=zscore_X,
                                            score_insample_orig=score_insample_orig)

            else:
                print("Error: unrecognizable model")
                raise SystemExit

            scores_all = pd.concat([scores_all, scores], ignore_index=True)
            best_params_all = pd.concat([best_params_all, best_params], ignore_index=True)

            scores_all.to_hdf(scores_path, key='p', mode='w')
            best_params_all.to_hdf(best_params_path, key='p', mode='w')

    if plot_scores:
        plot_all_scores(scores_all, title_prefix="Classification - ", save_folder=result_dir)


DATA_DIR = "data/"
# DATA_DIR = "/Users/hasnainmamdani/Academics/McGill/thesis/data/"

X = get_PC_data(DATA_DIR)
Y = get_patient_scores(DATA_DIR)

print("X.shape", X.shape, "Y.shape", Y.shape)

mixup_alphas = [0.3]
mixup_mul_factors = [5, 10]

result_dir = "results/3-"

run_with_mixup(X, Y, "rf", mixup_alphas, mixup_mul_factors, without_mixup=False, result_dir=result_dir,
               use_extreme_all_perc=0.4, plot_scores=False)

