import argparse
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

# To ensure reproducibility
random.seed(39)
np.random.seed(39)


def get_PC_data(data_dir):
    # Calculate 100 PC components
    if not os.path.isfile(data_dir + 'binary_imgs_pc_100.npy'):
        imgs = np.load(data_dir + "binary_imgs.npy")  # ensure they are there
        print("Loaded binary images. Size:", imgs.shape)

        pca = PCA(n_components=100, copy=False)
        X_pc = pca.fit_transform(imgs)
        from joblib import dump
        dump(pca, data_dir + 'pca100.joblib')
        np.save(data_dir + 'binary_imgs_pc_100.npy', X_pc)
        print("Created/saved 100 PC components. Size:", X_pc.shape)

    else:
        X_pc = np.load(data_dir + 'binary_imgs_pc_100.npy')
        print("Loaded 100 PC components. Size:", X_pc.shape)

    return X_pc


def get_atlas_lesion_load_matrix(data_dir):
    # load lesion load matrix
    return np.load(data_dir + "combined_lesions_load_matrix.npy")


def get_patient_idx_with_no_previous_lesions(data_dir):

    patient_select_df = pd.read_excel(data_dir + 'stroke-dataset/HallymBundang_MultiOutcome_25062020_selection_nopass.xlsx',
                                      sep=',', skipinitialspace=True)

    idx = np.array(patient_select_df["ID"]) - 1001  # IDs start with 1001

    idx = idx[:1154]  # corresponding brain images beyond this index are not available

    return idx


def get_patient_scores(data_dir, language_only=False, filter_idx=None):

    if language_only:

        patient_select_df = pd.read_excel(data_dir + 'stroke-dataset/HallymBundang_MultiOutcome_25062020_selection_nopass.xlsx',
                               sep=',', skipinitialspace=True)

        language_scores = np.array(patient_select_df[["SVLT_immediate_recall", "SVLT_delayed_recall", "SVLT_recognition"]])

        return language_scores[:1154]

    if not os.path.isfile(data_dir + 'cognitive_scores.npy'):

        patient_df = pd.read_excel(data_dir + 'stroke-dataset/HallymBundang_database_Bzdok_21082019.xlsx',
                                   sep=',', skipinitialspace=True)

        score_columns = ["MMSE_total_zscore", "BostonNamingTest_zscore", "ReyComplexFigureTestCopy_zscore",
                     "Seoul_Verbal_Learning_Test_immediate_recall_total_zscore", "TMT_A_Time_zscore_neg", "TMT_B_Time_zscore_neg"]

        scores_df = patient_df[score_columns]

        # preprocess variables - fill in the missing values by random sampling of valid values (simple random imputation)
        for score_name in scores_df.columns:
            invalid_entries = scores_df[scores_df[score_name] == 999.0].index
            print('Score name:', score_name, ', \tInvalid values replaced:', len(invalid_entries))
            valid_values = scores_df[score_name][scores_df[score_name] != 999].values
            scores_df[score_name].loc[invalid_entries] = random.choices(list(valid_values), k=len(invalid_entries))

        scaler_Y = StandardScaler()
        scores = scaler_Y.fit_transform(scores_df)

        np.save(data_dir + 'cognitive_scores.npy', scores)

        print("Processed and saved cognitive scores. Size:", scores.shape)

    else:
        scores = np.load(data_dir + 'cognitive_scores.npy')
        print("Loaded cognitive scores. Size:", scores.shape)

    if filter_idx is not None:
        return scores[filter_idx]

    return scores


# common library

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


def calculate_regression_metrics(est_cv, X_train, X_test, Y_train, Y_test, model_name, i_fold, score_insample_orig=False,
                                 X_train_orig=None, Y_train_orig=None, score_domain=None):

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


def run_regression(X, Y, estimator, grid, model_name, random_search_cv=False, n_jobs=-1, random_iter=16,
                   pca_fold=False, nn=False, callbacks=None, print_best_params=True,
                   mixup=False, mixup_alpha=0.1, mixup_mul_factor=5,
                   log_X=False, zscore_X=False, zscore_Y=False, score_insample_orig=True):

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

        X_train_upd = np.copy(X_train)
        Y_train_upd = np.copy(Y_train)
        X_test_upd = X_test
        Y_test_upd = Y_test

        if mixup:
            X_train_tmp, Y_train_tmp = mixup_data(X_train_upd, Y_train_upd, alpha=mixup_alpha, mul_factor=mixup_mul_factor)
            X_train_upd, Y_train_upd = X_train_tmp, Y_train_tmp

        if log_X:
            X_train_tmp = np.log(1 + X_train_upd)
            X_train_upd = X_train_tmp
            X_test_tmp = np.log(1 + X_test_upd)
            X_test_upd = X_test_tmp

        if zscore_X:
            scaler_X = StandardScaler()
            X_train_tmp = scaler_X.fit_transform(X_train_upd)
            X_train_upd = X_train_tmp
            X_test_tmp = scaler_X.transform(X_test_upd)
            X_test_upd = X_test_tmp

        if zscore_Y:
            scaler_Y = StandardScaler()
            Y_train_tmp = scaler_Y.fit_transform(Y_train_upd)
            Y_train_upd = Y_train_tmp
            Y_test_tmp = scaler_Y.transform(Y_test_upd)
            Y_test_upd = Y_test_tmp

        kfold_inner = KFold(n_splits=5, random_state=31)

        if random_search_cv:
            est_cv = RandomizedSearchCV(estimator=estimator, param_distributions=grid, n_jobs=n_jobs,
                                        cv=kfold_inner, random_state=39, n_iter=random_iter)
        else:
            est_cv = GridSearchCV(estimator=estimator, param_grid=grid, n_jobs=n_jobs, cv=kfold_inner)

        print('\nFold-' + str(i_fold) + '\n')

        if nn:
            est_cv.fit(X_train_upd, Y_train_upd, nn=True, callbacks=callbacks)

        else:
            est_cv.fit(X_train_upd, Y_train_upd)

        if score_insample_orig:
            if log_X:
                X_train_tmp = np.log(1 + X_train)
                X_train = X_train_tmp
            if zscore_X:
                X_train_tmp = scaler_X.transform(X_train)
                X_train = X_train_tmp
            if zscore_Y:
                Y_train_tmp = scaler_Y.transform(Y_train)
                Y_train = Y_train_tmp

            scores_fold = calculate_regression_metrics(est_cv, X_train_upd, X_test_upd, Y_train_upd, Y_test_upd, model_name, i_fold,
                                                       True, X_train, Y_train)

        else:
            scores_fold = calculate_regression_metrics(est_cv, X_train_upd, X_test_upd, Y_train_upd, Y_test_upd, model_name, i_fold)

        scores.extend(scores_fold)

        best_params.append([model_name, i_fold, est_cv.best_params_])

        if print_best_params:
            print("Best params:", best_params[-1])

        i_fold += 1

    scores_df = pd.DataFrame(scores, columns=["Model", "Domain", "Metric", "Score type", "Fold", "Score"])
    best_params_df = pd.DataFrame(best_params, columns=["Model", "Fold", "Best params"])

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

    print(scores_df[(scores_df["Score type"] == "Out-of-sample") & (scores_df["Metric"] == "R2")])

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


def multitask_ridge(X, Y, mixup, mixup_alpha=None, mixup_mul_factor=None, log_X=False, zscore_X=False, zscore_Y=False,
                    score_insample_orig=True):
    print('Performing Multitask regression (5-fold nested CV)')
    estimator = MultiTaskElasticNet(l1_ratio=0.001, max_iter=10000)

    if mixup:
        if mixup_mul_factor == 5:
            alpha = np.linspace(0.001, 0.1, 100)
        elif mixup_mul_factor == 10:
            alpha = np.linspace(0.0001, 0.01, 100)
        else:
            alpha = np.linspace(0.00001, 0.001, 100)

    else:
        alpha = np.linspace(0.1, 10, 100)

    grid = {"alpha": alpha}

    return run_regression(X, Y, estimator, grid, "Multitask Ridge",
                          mixup=mixup, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                          log_X=log_X, zscore_X=zscore_X, zscore_Y=zscore_Y, score_insample_orig=score_insample_orig)


def random_forest(X, Y, mixup, mixup_alpha=None, mixup_mul_factor=None, log_X=False, zscore_X=False, zscore_Y=False,
                  score_insample_orig=True):
    print('Performing Random Forest regression (5-fold nested CV)')
    estimator = RandomForestRegressor(random_state=39)

    n_estimators = [2250, 2500, 2750, 3000, 3250, 3500]
    max_depth = [10, 50, 100, 150, 200, 250, 275, 300, 350, 400, None]
    min_samples_split = [2, 3, 4, 5, 6, 7]
    min_samples_leaf = [1, 2, 3]
    max_samples = [0.5, 0.7, 0.8, 0.9, None]  # None turns out to be the best

    grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'max_samples': max_samples}

    return run_regression(X, Y, estimator, grid, "Random Forest", random_search_cv=True, random_iter=16,
                          mixup=mixup, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                          log_X=log_X, zscore_X=zscore_X, zscore_Y=zscore_Y, score_insample_orig=score_insample_orig)


def pls(X, Y, mixup, mixup_alpha=None, mixup_mul_factor=None, log_X=False, zscore_X=False, zscore_Y=False,
        score_insample_orig=True):
    print('Performing PLS regression (5-fold nested CV)')
    estimator = PLSRegression()
    n_components = np.linspace(1, Y.shape[1], Y.shape[1], dtype=int)  # Y.shape[1]=6
    grid = {'n_components': n_components}

    return run_regression(X, Y, estimator, grid, "PLS",
                          mixup=mixup, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                          log_X=log_X, zscore_X=zscore_X, zscore_Y=zscore_Y, score_insample_orig=score_insample_orig)


def cca(X, Y, mixup, mixup_alpha=None, mixup_mul_factor=None, log_X=False, zscore_X=False, zscore_Y=False,
        score_insample_orig=True):
    print('Performing CCA (5-fold nested CV)')
    estimator = CCA()
    n_components = np.linspace(1, Y.shape[1], Y.shape[1], dtype=int)  # Y.shape[1]=6
    grid = {'n_components': n_components}
    print("CCA")

    return run_regression(X, Y, estimator, grid, "CCA",
                          mixup=mixup, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                          log_X=log_X, zscore_X=zscore_X, zscore_Y=zscore_Y, score_insample_orig=score_insample_orig)


def run_with_mixup(X, Y, model, mixup_alphas, mixup_mul_factors, without_mixup=True, result_dir="", plot_scores=False,
                   log_X=False, zscore_X=False, zscore_Y=False, score_insample_orig=True):

    print("log_X=", log_X, "zscore_X=", zscore_X, "zscore_Y=", zscore_Y, "mixup_alphas=", mixup_alphas,
          "mixup_mul_factors=", mixup_mul_factors, "without_mixup=", without_mixup, "result_dir=", result_dir,
          "score_insample_orig=", score_insample_orig)

    scores_all = pd.DataFrame()
    best_params_all = pd.DataFrame()

    p_str = "-x" + str(mixup_mul_factors[0]) + "-" + str(mixup_alphas[0])
    p_str += "-log-after" if log_X else ""
    scores_path = result_dir + "regression-multioutput-mixup-" + model + p_str + ".h5"
    best_params_path = result_dir + "best-params-regression-multioutput-mixup-" + model + p_str + ".h5"
    print(scores_path)

    if without_mixup:
        # also get results without mixup (baseline)
        print("Performing regression without mixup")

        if model == "mridge":
            scores_no_mixup, best_params_no_mixup = multitask_ridge(X, Y, mixup=False, log_X=log_X, zscore_X=zscore_X,
                                                          zscore_Y=zscore_Y, score_insample_orig=score_insample_orig)

        elif model == "rf":
            scores_no_mixup, best_params_no_mixup = random_forest(X, Y, mixup=False, log_X=log_X, zscore_X=zscore_X,
                                                          zscore_Y=zscore_Y, score_insample_orig=score_insample_orig)

        elif model == "pls":
            scores_no_mixup, best_params_no_mixup = pls(X, Y, mixup=False, log_X=log_X, zscore_X=zscore_X,
                                                          zscore_Y=zscore_Y, score_insample_orig=score_insample_orig)

        elif model == "cca":
            scores_no_mixup, best_params_no_mixup = cca(X, Y, mixup=False, log_X=log_X, zscore_X=zscore_X,
                                                          zscore_Y=zscore_Y, score_insample_orig=score_insample_orig)

        else:
            print("Error: unrecognizable model")
            raise SystemExit

        scores_all = pd.concat([scores_all, scores_no_mixup], ignore_index=True)
        best_params_all = pd.concat([best_params_all, best_params_no_mixup], ignore_index=True)

    for mixup_mul_factor in mixup_mul_factors:

        for mixup_alpha in mixup_alphas:

            print("\n\nData mul factor:", str(mixup_mul_factor)+"x,", "Mixup alpha:", mixup_alpha)

            if model == "mridge":
                scores, best_params = multitask_ridge(X, Y, mixup=True, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                                            log_X=log_X, zscore_X=zscore_X, zscore_Y=zscore_Y, score_insample_orig=score_insample_orig)

            elif model == "rf":
                scores, best_params = random_forest(X, Y, mixup=True, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                                            log_X=log_X, zscore_X=zscore_X, zscore_Y=zscore_Y, score_insample_orig=score_insample_orig)

            elif model == "pls":
                scores, best_params = pls(X, Y, mixup=True, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                                            log_X=log_X, zscore_X=zscore_X, zscore_Y=zscore_Y, score_insample_orig=score_insample_orig)

            elif model == "cca":
                scores, best_params = cca(X, Y, mixup=True, mixup_alpha=mixup_alpha, mixup_mul_factor=mixup_mul_factor,
                                            log_X=log_X, zscore_X=zscore_X, zscore_Y=zscore_Y, score_insample_orig=score_insample_orig)

            else:
                print("Error: unrecognizable model")

            scores_all = pd.concat([scores_all, scores], ignore_index=True)
            best_params_all = pd.concat([best_params_all, best_params], ignore_index=True)

    scores_all.to_hdf(scores_path, key='p', mode='w')
    best_params_all.to_hdf(best_params_path, key='p', mode='w')

    # if plot_scores:
    #     plot_all_scores(scores_all, title_prefix="Regression - ", save_folder=result_dir)


use_pc100 = False  # False for atlas ROI lesion load matrix

parser = argparse.ArgumentParser()
parser.add_argument("--mixup-alpha", nargs="*", type=float, default=[0.01])
parser.add_argument("--mixup-mul-factor", nargs="*", type=int, default=[5])
parser.add_argument("--data-dir", default="/Users/hasnainmamdani/Academics/McGill/thesis/data/")
parser.add_argument("--result-dir", default="results/" + ("pc100_" if use_pc100 else "atlas_llm_"))
parser.add_argument("--model", default="pls")
parser.add_argument("--language_scores_only", default=True)
args = parser.parse_args()
print(args)

DATA_DIR = args.data_dir

filter_idx = get_patient_idx_with_no_previous_lesions(DATA_DIR)

if use_pc100:
    X = get_PC_data(DATA_DIR)

else:
    X = get_atlas_lesion_load_matrix(DATA_DIR)
    # X = np.log(1 + llm) # for mixup after log

X = X[filter_idx]
Y = get_patient_scores(DATA_DIR, args.language_scores_only, filter_idx)

print("X.shape", X.shape, "Y.shape", Y.shape)


if args.language_scores_only:
    SCORE_DOMAINS = ['SVLT Immediate Recall', 'SVLT Delayed Recall', 'SVLT Recognition']
else:
    SCORE_DOMAINS = ['Global Cognition', 'Language', 'Visuospatial Functioning', 'Memory',
                     'Information Processing Speed', 'Executive Functioning']

mixup_alphas = args.mixup_alpha
mixup_mul_factors = args.mixup_mul_factor

without_mixup = True if (mixup_alphas[0] == 0.01 and mixup_mul_factors[0] == 5) else False

run_with_mixup(X, Y, args.model, mixup_alphas, mixup_mul_factors, without_mixup=without_mixup, result_dir=args.result_dir,
               log_X=not use_pc100, zscore_X=True, zscore_Y=True)
