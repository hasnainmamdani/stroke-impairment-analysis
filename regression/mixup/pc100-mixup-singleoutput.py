import os
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
import math

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


## common library

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


SCORE_DOMAINS = ['Global Cognition', 'Language', 'Visuospatial Functioning', 'Memory',\
                 'Information Processing Speed', 'Executive Functioning']


def calculate_regression_metrics(est_cv, X_train, X_test, Y_train, Y_test, model_name, i_fold, score_domain=None,
                                 score_insample_orig=False, X_train_orig=None, Y_train_orig=None):

    scores = []

    Y_predicted_train = est_cv.predict(X_train)
    Y_predicted_test = est_cv.predict(X_test)

    train_rsq = r2_score(Y_train, Y_predicted_train, multioutput='raw_values')
    test_rsq = r2_score(Y_test, Y_predicted_test, multioutput='raw_values')

    train_mae = mean_absolute_error(Y_train, Y_predicted_train, multioutput='raw_values')
    test_mae = mean_absolute_error(Y_test, Y_predicted_test, multioutput='raw_values')

    if score_insample_orig:
        Y_predicted_train_orig = est_cv.predict(X_train_orig)
        train_orig_rsq = r2_score(Y_train_orig, Y_predicted_train_orig, multioutput='raw_values')
        train_orig_mae = mean_absolute_error(Y_train_orig, Y_predicted_train_orig, multioutput='raw_values')

    if score_domain is None:  # multioutput model

        for i in range(len(SCORE_DOMAINS)):
            scores.append([model_name, SCORE_DOMAINS[i], "R2", "In-sample", i_fold, train_rsq[i]])
            scores.append([model_name, SCORE_DOMAINS[i], "R2", "Out-of-sample", i_fold, test_rsq[i]])
            scores.append([model_name, SCORE_DOMAINS[i], "MAE", "In-sample", i_fold, train_mae[i]])
            scores.append([model_name, SCORE_DOMAINS[i], "MAE", "Out-of-sample", i_fold, test_mae[i]])
            if score_insample_orig:
                scores.append([model_name, SCORE_DOMAINS[i], "R2", "In-sample (original)", i_fold, train_orig_rsq[i]])
                scores.append([model_name, SCORE_DOMAINS[i], "MAE", "In-sample (original)", i_fold, train_orig_mae[i]])

    else:

        scores.append([model_name, score_domain, "R2", "In-sample", i_fold, train_rsq[0]])
        scores.append([model_name, score_domain, "R2", "Out-of-sample", i_fold, test_rsq[0]])
        scores.append([model_name, score_domain, "MAE", "In-sample", i_fold, train_mae[0]])
        scores.append([model_name, score_domain, "MAE", "Out-of-sample", i_fold, test_mae[0]])
        if score_insample_orig:
            scores.append([model_name, score_domain, "R2", "In-sample (original)", i_fold, train_orig_rsq[0]])
            scores.append([model_name, score_domain, "MAE", "In-sample (original)", i_fold, train_orig_mae[0]])

    return scores


# get top and bottom p% of the data based on target variable and truncate the date in between
def get_extremes(X, y, p):
    sort_idx = np.argsort(y)

    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    extremes_idx = list(range(0, math.floor(X_sorted.shape[0] * p))) + \
                   list(range(math.ceil(X_sorted.shape[0] * (1 - p)), len(sort_idx)))

    return X_sorted[extremes_idx], y_sorted[extremes_idx]


def run_regression(X, Y, estimator, grid, model_name, random_search_cv=False, n_jobs=-1,
                   random_iter=20, nn=False, callbacks=None, print_best_params=True,
                   multioutput=False, mixup=False, mixup_alpha=0.1, mixup_mul_factor=5,
                   use_extreme_perc=None, score_insample_orig=True):

    if use_extreme_perc is not None and multioutput:
        print("Using extreme values for fitting is not supported with multioutput models")
        raise SystemExit()

    scores = []
    best_params = []

    kfold_outer = KFold(n_splits=5, shuffle=True, random_state=39)

    i_fold = 1

    print('\n% Data used - ' + ("100" if use_extreme_perc is None else str(use_extreme_perc*2*100)) + '\n')

    for train, test in kfold_outer.split(X):

        X_train = X[train]
        X_test = X[test]
        Y_train = Y[train]
        Y_test = Y[test]

        if score_insample_orig and (use_extreme_perc is not None or mixup):
            X_train_orig = np.copy(X_train)
            Y_train_orig = np.copy(Y_train)

        kfold_inner = KFold(n_splits=5)

        if random_search_cv:
            est_cv = RandomizedSearchCV(estimator=estimator, param_distributions=grid, n_jobs=n_jobs,
                                        cv=kfold_inner, random_state=39, n_iter=random_iter)
        else:
            est_cv = GridSearchCV(estimator=estimator, param_grid=grid, n_jobs=n_jobs,
                                  cv=kfold_inner)

        print('\nFold-' + str(i_fold) + '\n')

        if multioutput:

            if mixup:
                X_train_tmp, Y_train_tmp = mixup_data(X_train, Y_train, alpha=mixup_alpha, mul_factor=mixup_mul_factor)  # two-step to avoid internal bugs
                X_train, Y_train = X_train_tmp, Y_train_tmp

            if nn:
                est_cv.fit(X_train, Y_train, nn=True, callbacks=callbacks)

            else:
                est_cv.fit(X_train, Y_train)

            # note the in-sample results here will for the 'modified' X_train
            scores_fold = calculate_regression_metrics(est_cv, X_train, X_test, Y_train, Y_test,
                                                       model_name, i_fold)

            scores.extend(scores_fold)

            best_params.append([model_name, i_fold, est_cv.best_params_])

            if print_best_params:
                print("Best params:", best_params[-1])

        else:  # single output models

            for i_domain in range(Y.shape[1]):

                print(SCORE_DOMAINS[i_domain])

                y_train = Y_train[:, i_domain]
                y_test = Y_test[:, i_domain]

                X_train_upd = np.copy(X_train)
                y_train_upd = np.copy(y_train)

                if use_extreme_perc is not None:  # take only top/bottom p% and truncate the middle
                    X_train_tmp, y_train_tmp = get_extremes(X_train_upd, y_train_upd, use_extreme_perc)  # two-step to avoid internal bugs
                    X_train_upd, y_train_upd = X_train_tmp, y_train_tmp

                # np.save("test/y_train_extreme_"+str(i_fold)+"_"+str(i_domain), y_train_upd)

                if mixup:
                    X_train_tmp, y_train_tmp = mixup_data(X_train_upd, y_train_upd, alpha=mixup_alpha, mul_factor=mixup_mul_factor)  # two-step to avoid internal bugs
                    X_train_upd, y_train_upd = X_train_tmp, y_train_tmp

                # to debug
                # np.save("test/X_train_"+str(i_fold)+"_"+str(i_domain), X_train)
                # np.save("test/X_train_updated_"+str(i_fold)+"_"+str(i_domain), X_train_upd)
                # np.save("test/y_train_"+str(i_fold)+"_"+str(i_domain), y_train)
                # np.save("test/y_train_updated_"+str(i_fold)+"_"+str(i_domain), y_train_upd)

                if nn:
                    est_cv.fit(X_train_upd, y_train_upd, nn=True, callbacks=callbacks)

                else:
                    est_cv.fit(X_train_upd, y_train_upd)

                if score_insample_orig and (use_extreme_perc is not None or mixup):
                    scores_fold = calculate_regression_metrics(est_cv, X_train_upd, X_test, y_train_upd, y_test, model_name, i_fold,
                                                               SCORE_DOMAINS[i_domain], score_insample_orig,
                                                               X_train_orig, Y_train_orig[:, i_domain])

                else:
                    scores_fold = calculate_regression_metrics(est_cv, X_train_upd, X_test, y_train_upd, y_test, model_name, i_fold,
                                                           SCORE_DOMAINS[i_domain])

                scores.extend(scores_fold)

                best_params.append([model_name, SCORE_DOMAINS[i_domain], i_fold, est_cv.best_params_])

                if print_best_params:
                    print("Best params:", best_params[-1])

        i_fold += 1

    scores_df = pd.DataFrame(scores, columns=["Model", "Domain", "Metric", "Score type", "Fold", "Score"])

    bp_cols = ["Model", "Fold", "Best params"] if multioutput else ["Model", "Domain", "Fold", "Best params"]
    best_params_df = pd.DataFrame(best_params, columns=bp_cols)

    if mixup:
        scores_df.insert(0, "alpha", mixup_alpha)  # or "\u03B1" - alpha symbol
        scores_df.insert(0, "Data mul factor", str(mixup_mul_factor)+"x")
        best_params_df.insert(0, "alpha", mixup_alpha)
        best_params_df.insert(0, "Data mul factor", str(mixup_mul_factor)+"x")
    else:
        scores_df.insert(0, "alpha", "No mixup")
        scores_df.insert(0, "Data mul factor", "No mixup")
        best_params_df.insert(0, "alpha", "No mixup")
        best_params_df.insert(0, "Data mul factor", "No mixup")

    if use_extreme_perc is not None:
        scores_df.insert(0, "% Data", use_extreme_perc*2*100)
        best_params_df.insert(0, "% Data", use_extreme_perc*2*100)
    else:
        scores_df.insert(0, "% Data", 100.0)
        best_params_df.insert(0, "% Data", 100.0)

    return scores_df, best_params_df


def plot_scores(score_df, metric, score_type, title="", save_folder="", hue_order=None):

    g = sns.catplot(x="Domain", y="Score", hue="alpha", col="Data mul factor", col_order=["No mixup", "5x", "10x", "50x"],
                    row="% Data", data=score_df, ci="sd", kind="bar", hue_order=["No mixup", 0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 50.0],
                    row_order=[100.0, 80.0, 60.0, 40.0, 20.0])

    g.set_axis_labels("", "Score (Mean and Standard Deviation across 5 CV folds)")

    for i, ax in enumerate(g.fig.axes):
        ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
        ax.axhline(0, color="black")

    g.fig.suptitle(title, y=1.08, fontsize=30)

    if metric == "R2" and score_type == "Out-of-sample":
        g.set(ylim=(-0.2, 0.2))

    if metric == "R2" and score_type == "In-sample":
        g.set(ylim=(-0.05, 1.0))

    # plt.show()
    plt.savefig(save_folder + title, bbox_inches="tight")


def plot_all_scores(score_df, title_prefix="", save_folder="", hue_order=None):

    for metric in score_df["Metric"].unique():

        for score_type in score_df["Score type"].unique():

            for model in score_df["Model"].unique():

                filtered_data = score_df[(score_df["Metric"] == metric) & (score_df["Score type"] == score_type)
                                         & (score_df["Model"] == model)]

                title = title_prefix + metric + " - " + score_type + " - " + model

                plot_scores(filtered_data, metric, score_type, title, save_folder, hue_order=hue_order)


def ridge(X, Y, mixup, mixup_alpha=None, mixup_mul_factor=None, use_extreme_perc=None, score_insample_orig=True):
    estimator = Ridge()

    # defining the hyperparameter range for vaious combinations (learnt from experience)
    if not mixup:
        if use_extreme_perc is None:
            alpha = np.linspace(1, 600001, 3001)
        else:
            alpha = np.linspace(1, 50001, 2001)

    else:
        if use_extreme_perc is None:
            alpha = np.linspace(1, 1501, 501)
        elif use_extreme_perc == 0.1:
            alpha = np.linspace(0.01, 1, 100)
        elif use_extreme_perc == 0.2:
            alpha = np.append(np.linspace(0.1, 0.9, 9), np.linspace(1, 100, 100))
        elif use_extreme_perc == 0.3:
            alpha = np.append(np.linspace(0.1, 0.9, 9), np.linspace(1, 200, 200))
        else:
            alpha = np.linspace(1, 502, 168)

    grid = {"alpha": alpha}
    print(grid)

    return run_regression(X, Y, estimator, grid, "Ridge", multioutput=False, mixup=mixup,
                          mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                          use_extreme_perc=use_extreme_perc, score_insample_orig=score_insample_orig)


def svr_rbf(X, Y, mixup, mixup_alpha=None, mixup_mul_factor=None, use_extreme_perc=None, score_insample_orig=True):
    estimator = SVR(kernel='rbf')
    grid = {"C": np.logspace(-1, 6, 8),
            "gamma": np.logspace(-3, 8, 12)}

    return run_regression(X, Y, estimator, grid, "SVR-RBF", multioutput=False, mixup=mixup,
                          mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                          use_extreme_perc=use_extreme_perc, score_insample_orig=score_insample_orig)


def random_forest(X, Y, mixup, mixup_alpha=None, mixup_mul_factor=None, use_extreme_perc=None, score_insample_orig=True):

    estimator = RandomForestRegressor(random_state=39)

    # n_estimators = [10, 25, 50, 100, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000]
    n_estimators = [10, 25, 50, 100, 150, 200, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000, 3500]
    max_features = ['auto', 'sqrt']
    # max_depth = [2, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 100, None] changed later
    max_depth = [2, 5, 7, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, None]
    # min_samples_split = [2, 5, 10, 15, 20, 25, 30, 40, 50, 70]
    min_samples_split = [2, 5, 10, 15, 20, 25, 30, 40, 50, 70, 80, 90]
    # min_samples_leaf = [1, 2, 4, 8, 12, 15, 18, 21]
    min_samples_leaf = [1, 2, 4, 8, 12, 15, 18, 21, 24]
    max_samples = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.7, 0.8, 0.9, None]

    grid = {'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'max_samples': max_samples}

    return run_regression(X, Y, estimator, grid, "Random Forest", random_search_cv=True, random_iter=24,
                          multioutput=False, mixup=mixup, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                          use_extreme_perc=use_extreme_perc, score_insample_orig=score_insample_orig)


def run_with_mixup(X, Y, model, mixup_alphas, mixup_mul_factors, without_mixup=True, result_dir="", plot_scores=False,
                   use_extreme_perc=None, score_insample_orig=True):

    scores_all = pd.DataFrame()
    best_params_all = pd.DataFrame()

    p_str = "-100.0" if use_extreme_perc is None else "-"+str(use_extreme_perc)
    scores_path = result_dir + "regression-singleoutput-pc100-mixup-" + model + p_str + ".h5"
    best_params_path = result_dir + "best-params-regression-singleoutput-pc100-mixup-" + model + p_str + ".h5"

    if without_mixup:
        # also get results without mixup (baseline)
        print("Performing regression without mixup")

        if model == "ridge":
            scores_no_mixup, best_params_no_mixup = ridge(X, Y, mixup=False,
                                                          use_extreme_perc=use_extreme_perc, score_insample_orig=score_insample_orig)

        elif model == "svr-rbf":
            scores_no_mixup, best_params_no_mixup = svr_rbf(X, Y, mixup=False,
            use_extreme_perc=use_extreme_perc, score_insample_orig=score_insample_orig)

        elif model == "rf":
            scores_no_mixup, best_params_no_mixup = random_forest(X, Y, mixup=False,
            use_extreme_perc=use_extreme_perc, score_insample_orig=score_insample_orig)

        else:
            print("Error: unrecognizable model")

        scores_all = pd.concat([scores_all, scores_no_mixup], ignore_index=True)
        best_params_all = pd.concat([best_params_all, best_params_no_mixup], ignore_index=True)

        scores_all.to_hdf(scores_path, key='p', mode='w')
        best_params_all.to_hdf(best_params_path, key='p', mode='w')

    for mixup_mul_factor in mixup_mul_factors:

        for mixup_alpha in mixup_alphas:

            print("\n\nData mul factor:", str(mixup_mul_factor)+"x,", "Mixup alpha:", mixup_alpha)

            if model == "ridge":
                scores, best_params = ridge(X, Y, mixup=True, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                                            use_extreme_perc=use_extreme_perc, score_insample_orig=score_insample_orig)

            elif model == "svr-rbf":
                scores, best_params = svr_rbf(X, Y, mixup=True, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                                              use_extreme_perc=use_extreme_perc, score_insample_orig=score_insample_orig)

            elif model == "rf":
                scores, best_params = random_forest(X, Y, mixup=True, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                                                    use_extreme_perc=use_extreme_perc, score_insample_orig=score_insample_orig)

            else:
                print("Error: unrecognizable model")
                raise SystemExit

            scores_all = pd.concat([scores_all, scores], ignore_index=True)
            best_params_all = pd.concat([best_params_all, best_params], ignore_index=True)

            scores_all.to_hdf(scores_path, key='p', mode='w')
            best_params_all.to_hdf(best_params_path, key='p', mode='w')

    if plot_scores:
        plot_all_scores(scores_all, title_prefix="Regression - ", save_folder=result_dir)


DATA_DIR = "data/"
# DATA_DIR = "/Users/hasnainmamdani/Academics/McGill/thesis/data/"

X = get_PC_data(DATA_DIR)
Y = get_patient_scores(DATA_DIR)

print("X.shape", X.shape, "Y.shape", Y.shape)

mixup_alphas = [0.01, 0.1, 0.3, 1.0, 3.0, 10.0, 50.0]
mixup_mul_factors = [5, 10]

result_dir = "results/"

run_with_mixup(X, Y, "ridge", mixup_alphas, mixup_mul_factors, without_mixup=True, result_dir=result_dir, use_extreme_perc=0.4)
# run_with_mixup(X, Y, "svr-rbf", mixup_alphas, mixup_mul_factors, without_mixup=True, result_dir=result_dir, use_extreme_perc=None)
# run_with_mixup(X, Y, "rf", mixup_alphas, mixup_mul_factors, without_mixup=False, result_dir=result_dir, use_extreme_perc=0.4)
