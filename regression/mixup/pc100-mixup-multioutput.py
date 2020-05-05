import os
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import MultiTaskElasticNet
from sklearn.cross_decomposition import PLSRegression, CCA
from sklearn.model_selection import GridSearchCV, KFold, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(39)

#To ensure reproducibility
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

    mixed_X = np.empty((n*(mul_factor-1), X.shape[1]))
    mixed_Y = np.empty((n*(mul_factor-1), Y.shape[1]))

    for i in range(mul_factor-1):

        # sample more than needed as some will be filtered out
        lam = np.random.beta(alpha, alpha, size=round(n*2))

        # original data vectors will be concatenated later
        lam = lam[(lam!=0.0) & (lam!=1.0)][:n][:, None]  # shape nx1

        shuffle_idx = rs.choice(np.arange(n), n, replace=False)

        mixed_X[i*n : (i+1)*n] = lam * X + (1 - lam) * X[shuffle_idx, :]
        mixed_Y[i*n : (i+1)*n] = lam * Y + (1 - lam) * Y[shuffle_idx, :]

    # concatenate original data vectors
    mixed_X = np.append(mixed_X, X, axis=0)
    mixed_Y = np.append(mixed_Y, Y, axis=0)

    return mixed_X, mixed_Y


SCORE_DOMAINS = ['Global Cognition', 'Language', 'Visuospatial Functioning', 'Memory',\
                 'Information Processing Speed', 'Executive Functioning']


def calculate_regression_metrics(est_cv, X_train, X_test, y_train, y_test, model_name, i_fold, score_domain=None):

    scores = []

    y_predicted_train = est_cv.predict(X_train)
    y_predicted_test = est_cv.predict(X_test)

    train_rsq = r2_score(y_train, y_predicted_train, multioutput='raw_values')
    test_rsq = r2_score(y_test, y_predicted_test, multioutput='raw_values')

    train_mae = mean_absolute_error(y_train, y_predicted_train, multioutput='raw_values')
    test_mae = mean_absolute_error(y_test, y_predicted_test, multioutput='raw_values')

    if score_domain is None: # multioutput model

        for i in range(len(SCORE_DOMAINS)):
            scores.append([model_name, SCORE_DOMAINS[i], "R2", "In-sample", i_fold, train_rsq[i]])
            scores.append([model_name, SCORE_DOMAINS[i], "R2", "Out-of-sample", i_fold, test_rsq[i]])
            scores.append([model_name, SCORE_DOMAINS[i], "MAE", "In-sample", i_fold, train_mae[i]])
            scores.append([model_name, SCORE_DOMAINS[i], "MAE", "Out-of-sample", i_fold, test_mae[i]])

    else:

        scores.append([model_name, score_domain, "R2", "In-sample", i_fold, train_rsq])
        scores.append([model_name, score_domain, "R2", "Out-of-sample", i_fold, test_rsq])
        scores.append([model_name, score_domain, "MAE", "In-sample", i_fold, train_mae])
        scores.append([model_name, score_domain, "MAE", "Out-of-sample", i_fold, test_mae])

    return scores


def run_regression(X, Y, estimator, grid, model_name, random_search_cv=False, n_jobs=-1, random_iter=20,
                   pca_fold=False, nn=False, callbacks=None, print_best_params=True,
                   multioutput=False, mixup=False, mixup_alpha=0.1, mixup_mul_factor=10):

    scores = []
    best_params = []

    kfold_outer = KFold(n_splits=5, shuffle=True, random_state=39)

    i_fold = 1

    for train, test in kfold_outer.split(X):

        X_train = X[train]
        X_test = X[test]

        if pca_fold: # to do PCA only on the training data for each fold
            pca = PCA(n_components=100)
            X_train_tmp = pca.fit_transform(X_train)
            X_test_tmp = pca.transform(X_test)
            X_train = X_train_tmp
            X_test = X_test_tmp

        Y_train = Y[train]
        Y_test = Y[test]

        if mixup:
            X_train_tmp, Y_train_tmp = mixup_data(X_train, Y_train, alpha=mixup_alpha, mul_factor=mixup_mul_factor)
            X_train, Y_train = X_train_tmp, Y_train_tmp

        kfold_inner = KFold(n_splits=5)

        if random_search_cv:
            est_cv = RandomizedSearchCV(estimator=estimator, param_distributions=grid, n_jobs=n_jobs,
                                        cv=kfold_inner, random_state=39, n_iter=random_iter)
        else:
            est_cv = GridSearchCV(estimator=estimator, param_grid=grid, n_jobs=n_jobs,
                                  cv=kfold_inner)


        print('\nFold-' + str(i_fold) + '\n')

        if multioutput:

            if nn:
                est_cv.fit(X_train, Y_train, nn=True, callbacks=callbacks)

            else:
                est_cv.fit(X_train, Y_train)
            scores_fold = calculate_regression_metrics(est_cv, X_train, X_test, Y_train, Y_test,
                                                       model_name, i_fold)

            scores.extend(scores_fold)

            best_params.append([model_name, i_fold, est_cv.best_params_])

            if print_best_params:
                print("Best params:", best_params[-1])

        else: # single output models

            for i_domain in range(Y.shape[1]):

                print(SCORE_DOMAINS[i_domain])

                y_train = Y_train[:, i_domain]
                y_test = Y_test[:, i_domain]


                if nn:
                    est_cv.fit(X_train, y_train, nn=True, callbacks=callbacks)

                else:
                    est_cv.fit(X_train, y_train)

                scores_fold = calculate_regression_metrics(est_cv, X_train, X_test, y_train, y_test,
                                                           model_name, i_fold, SCORE_DOMAINS[i_domain])

                scores.extend(scores_fold)

                best_params.append([model_name, SCORE_DOMAINS[i_domain], i_fold, est_cv.best_params_])

                if print_best_params:
                    print("Best params:", best_params[-1])

        i_fold += 1

    scores_df = pd.DataFrame(scores, columns=["Model", "Domain", "Metric", "Score type", "Fold", "Score"])

    bp_cols = ["Model", "Fold", "Best params"] if multioutput else ["Model", "Domain", "Fold", "Best params"]
    best_params_df = pd.DataFrame(best_params, columns=bp_cols)

    if mixup:
        scores_df.insert(0, "alpha", mixup_alpha) # or "\u03B1" - alpha symbol
        scores_df.insert(0, "Data mul factor", str(mixup_mul_factor)+"x")
        best_params_df.insert(0, "alpha", mixup_alpha)
        best_params_df.insert(0, "Data mul factor", str(mixup_mul_factor)+"x")
    else:
        scores_df.insert(0, "alpha", "No mixup")
        scores_df.insert(0, "Data mul factor", "No mixup")
        best_params_df.insert(0, "alpha", "No mixup")
        best_params_df.insert(0, "Data mul factor", "No mixup")

    return scores_df, best_params_df


def plot_scores(score_df, title="", save_folder="", hue_order=None):

    g = sns.catplot(x="Domain", y="Score", hue="alpha", col="Data mul factor",
                    data=score_df, ci="sd", kind="bar", hue_order=hue_order) # col_order=["No mixup", "5x", "10x", "50x"]

    g.set_axis_labels("", "Score (Mean and Standard Deviation across 5 CV folds)")

    for i, ax in enumerate(g.fig.axes):
        ax.set_xticklabels(ax.get_xticklabels(), rotation=65)
        ax.axhline(0, color="black")

    g.fig.suptitle(title, y=1.08, fontsize=30)
    # plt.show()
    plt.savefig(save_folder + title, bbox_inches="tight")


def plot_all_scores(score_df, title_prefix="", save_folder="", hue_order=None):

    for metric in score_df["Metric"].unique():

        for score_type in score_df["Score type"].unique():

            filtered_data = score_df[(score_df["Metric"] == metric) & (score_df["Score type"] == score_type)]

            title = title_prefix + metric + " - " + score_type

            plot_scores(filtered_data, title, save_folder, hue_order=hue_order)


def multitask_ridge(X, Y, alpha, mixup=True, mixup_alpha=None, mixup_mul_factor=None):

    estimator = MultiTaskElasticNet(l1_ratio=0.001, max_iter=10000)
    grid = {"alpha": alpha}

    return run_regression(X, Y, estimator, grid, "Multitask Ridge", multioutput=True,
                          mixup=mixup, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor)


def run_multitask_ridge_mixup(X, Y, without_mixup=True, result_path="", plot_scores=True, plot_title_prefix=""):

    # first get results with no mixup
    scores_mr_all = pd.DataFrame()
    best_params_mr_all = pd.DataFrame()

    if without_mixup:
        # get results with no mixup
        ridge_alpha = np.concatenate(([0.1], np.linspace(1, 50, 50)), axis=0)
        scores_mr_no_mixup, best_params_mr_no_mixup = multitask_ridge(X, Y, ridge_alpha, mixup=False)

        scores_mr_all = pd.concat([scores_mr_all, scores_mr_no_mixup], ignore_index=True)
        best_params_mr_all = pd.concat([best_params_mr_all, best_params_mr_no_mixup], ignore_index=True)

    mixup_alphas = [0.1, 0.2]
    mixup_mul_factors = [5, 10]

    ridge_alphas = {
        5: [0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.14, 0.16, 0.18, 0.2, 0.3, 5],
        10: [0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.12, 0.2, 5],
        50: [0.0006, 0.0008, 0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.02, 5]
    }

    for mixup_mul_factor in mixup_mul_factors:

        for mixup_alpha in mixup_alphas:

            print("\n\nData mul factor:", str(mixup_mul_factor)+"x,", "Mixup alpha:", mixup_alpha)

            scores_mr, best_params_mr = multitask_ridge(X, Y, ridge_alphas[mixup_mul_factor], mixup=True,
                                                        mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor)

            scores_mr_all = pd.concat([scores_mr_all, scores_mr], ignore_index=True)
            best_params_mr_all = pd.concat([best_params_mr_all, best_params_mr], ignore_index=True)

            scores_mr_all.to_hdf(result_path + "regression-multioutput-pc100-mixup-mr.h5", key='p', mode='w')
            best_params_mr_all.to_hdf(result_path + "best-params-regression-multioutput-pc100-mixup-mr.h5", key='p', mode='w')

    if plot_scores:
        plot_all_scores(scores_mr_all, title_prefix=plot_title_prefix, save_folder=result_path)


def random_forest(X, Y, mixup, mixup_alpha=None, mixup_mul_factor=None):
    print('Performing Random Forest regression (5-fold nested CV)')
    estimator = RandomForestRegressor(random_state=39)

    n_estimators = [2250, 2500, 2750, 3000, 3250, 3500]
    max_depth = [10, 50, 100, 150, 200, 250, 275, 300, 350, 400, None]
    min_samples_split = [2, 3, 4, 5, 6, 7]
    min_samples_leaf = [1, 2, 3]
    max_samples = [None]  # None turns out to be the best

    grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_samples': max_samples}

    return run_regression(X, Y, estimator, grid, "Random Forest", random_search_cv=True, random_iter=16,
                          multioutput=True, mixup=mixup, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor)


def pls(X, Y, mixup, mixup_alpha=None, mixup_mul_factor=None):

    estimator = PLSRegression()
    n_components = np.linspace(1, Y.shape[1], Y.shape[1], dtype=int)  # Y.shape[1]=6
    grid = {'n_components': n_components}

    return run_regression(X, Y, estimator, grid, "PLS", multioutput=True,
                          mixup=mixup, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor)


def cca(X, Y, mixup, mixup_alpha=None, mixup_mul_factor=None):

    estimator = CCA()
    n_components = np.linspace(1, Y.shape[1], Y.shape[1], dtype=int)  # Y.shape[1]=6
    grid = {'n_components': n_components}

    return run_regression(X, Y, estimator, grid, "CCA", multioutput=True,
                          mixup=mixup, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor)


def run_with_mixup(X, Y, model, without_mixup=True, result_path="", plot_scores=True, plot_title_prefix=""):

    scores_all = pd.DataFrame()
    best_params_all = pd.DataFrame()

    if without_mixup:
        # get results with no mixup
        print("Performing regression without mixup")

        if model == "rf":
            scores_no_mixup, best_params_no_mixup = random_forest(X, Y, mixup=False)

        elif model == "pls":
            scores_no_mixup, best_params_no_mixup = pls(X, Y, mixup=False)

        elif model == "cca":
            scores_no_mixup, best_params_no_mixup = cca(X, Y, mixup=False)

        else:
            print("Error: unrecognizable model")

        scores_all = pd.concat([scores_all, scores_no_mixup], ignore_index=True)
        best_params_all = pd.concat([best_params_all, best_params_no_mixup], ignore_index=True)

    mixup_alphas = [0.01, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 2.0, 5.0]
    mixup_mul_factors = [5, 10]

    for mixup_mul_factor in mixup_mul_factors:

        for mixup_alpha in mixup_alphas:

            print("\n\nData mul factor:", str(mixup_mul_factor)+"x,", "Mixup alpha:", mixup_alpha)

            if model == "rf":
                scores, best_params = random_forest(X, Y, mixup=True, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor)

            elif model == "pls":
                scores, best_params = pls(X, Y, mixup=True, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor)

            elif model == "cca":
                scores, best_params = cca(X, Y, mixup=True, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor)

            else:
                print("Error: unrecognizable model")

            scores_all = pd.concat([scores_all, scores], ignore_index=True)
            best_params_all = pd.concat([best_params_all, best_params], ignore_index=True)

            scores_all.to_hdf(result_path + "regression-multioutput-pc100-mixup-" + model + ".h5", key='p', mode='w')
            best_params_all.to_hdf(result_path + "best-params-regression-multioutput-pc100-mixup-" + model + ".h5", key='p', mode='w')

    if plot_scores:
        plot_all_scores(scores_all, title_prefix=plot_title_prefix, save_folder=result_path)


DATA_DIR = "data/"
# DATA_DIR = "/Users/hasnainmamdani/Academics/McGill/thesis/data/"
result_path = "results/"

X = get_PC_data(DATA_DIR)
Y = get_patient_scores(DATA_DIR)

print("X.shape", X.shape, "Y.shape", Y.shape)

# run_multitask_ridge_mixup(X, Y, plot_title_prefix="Regression - Multitask Ridge - ", result_path=result_path)
# run_with_mixup(X, Y, model="rf", plot_title_prefix="Regression - Random Forest - ", result_path=result_path)
run_with_mixup(X, Y, model="pls", plot_title_prefix="Regression - PLS - ", result_path=result_path)
run_with_mixup(X, Y, model="cca", plot_title_prefix="Regression - CCA - ", result_path=result_path)
