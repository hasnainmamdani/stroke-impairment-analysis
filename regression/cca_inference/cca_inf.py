import argparse
import os
import numpy as np
import pandas as pd
import random
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import CCA
from nilearn.signal import clean
from scipy.stats import pearsonr
from matplotlib import pylab as plt
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
    """
    :param language_only: use language scores only
    :param filter_idx: For all cognitive scores. Language scores are already filtered.
    :return:
    """

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

        assert filter_idx is not None, "must filter stroke patients with prior stroke lesions"
        scores_df = scores_df.iloc[filter_idx].reset_index(drop=True)

        # preprocess variables - fill in the missing values by random sampling of valid values (simple random imputation)
        for score_name in scores_df.columns:
            invalid_entries = scores_df[scores_df[score_name] == 999.0].index
            print('Score name:', score_name, ', \tInvalid values replaced:', len(invalid_entries))
            valid_values = scores_df[score_name][scores_df[score_name] != 999].values
            scores_df[score_name].loc[invalid_entries] = random.choices(list(valid_values), k=len(invalid_entries))

        # these scores were already z-scored, hence better to re-zscore them now

        scaler_Y = StandardScaler()
        scores = scaler_Y.fit_transform(scores_df)

        np.save(data_dir + 'cognitive_scores.npy', scores)

        print("Processed and saved cognitive scores. Size:", scores.shape)

    else:
        scores = np.load(data_dir + 'cognitive_scores.npy')
        print("Loaded cognitive scores. Size:", scores.shape)

    return scores


def deconfound(data_dir, X):

    patient_select_df = pd.read_excel(data_dir + 'stroke-dataset/HallymBundang_MultiOutcome_25062020_selection_nopass.xlsx',
                                      sep=',', skipinitialspace=True)[:1154]

    age = StandardScaler().fit_transform(patient_select_df["Age"].values.reshape(-1, 1))
    age2 = age ** 2
    sex = pd.get_dummies(patient_select_df["Sex"]).values[:, 0].reshape(-1, 1)
    sex_x_age = sex * age
    sex_x_age2 = sex * age2
    edu = StandardScaler().fit_transform(patient_select_df["Education_years"].values.reshape(-1, 1))
    infarct_volume = StandardScaler().fit_transform(patient_select_df["Total_infarct_volume"].values.reshape(-1, 1))
    cohort = pd.get_dummies(patient_select_df["Cohort"]).values[:, 0].reshape(-1, 1)

    conf_mat = np.hstack([age, age2, sex, sex_x_age, sex_x_age2, edu, infarct_volume, cohort])

    X_conf = clean(X, confounds=conf_mat, detrend=False, standardize=False)

    return X_conf


def plot_cca_loadings(X, Y, data_dir, include_cerebellum_regions):
    n_keep = 3

    cca = CCA(n_components=n_keep, scale=False)
    cca.fit(X, Y)

    roi_names = np.load(data_dir + "combined_atlas_region_labels.npy")
    if not include_cerebellum_regions:
        roi_names = roi_names[np.r_[0:111, 145:193]]

    langauge_score_names = ['SVLT Immediate Recall', 'SVLT Delayed Recall', 'SVLT Recognition']

    for i_ccomp in range(n_keep):

        plt.figure(figsize=(50, 30))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.axhline(0, color="black")
        plt.title('Canonical component %i: Atlas regions' % (i_ccomp + 1))
        plt.bar(roi_names, cca.x_loadings_[:, i_ccomp])
        plt.savefig('cca_x_%iof%i' % (i_ccomp + 1, n_keep), bbox_inches='tight')

        plt.clf()
        plt.figure(figsize=(6, 5))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.axhline(0, color="black")
        plt.title('Canonical component %i: Language scores' % (i_ccomp + 1))
        plt.bar(langauge_score_names, cca.y_loadings_[:, i_ccomp])
        plt.savefig('cca_y_%iof%i' % (i_ccomp + 1, n_keep), bbox_inches='tight')


def permutation_test(X, Y):

    n_keep = 3
    n_permutations = 1000

    actual_cca = CCA(n_components=n_keep, scale=False)
    actual_cca.fit(X, Y)
    actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in zip(actual_cca.x_scores_.T, actual_cca.y_scores_.T)])

    perm_rs = np.random.RandomState(72)
    perm_Rs = []
    n_except = 0
    for i_iter in range(n_permutations):
        if i_iter % 50 == 0:
            print(i_iter + 1)

        Y_perm = np.array([perm_rs.permutation(sub_row) for sub_row in Y])

        # same procedure, only with permuted subjects on the right side
        try:
            perm_cca = CCA(n_components=n_keep, scale=False)

            # perm_inds = np.arange(len(Y_netmet))
            # perm_rs.shuffle(perm_inds)
            # perm_cca.fit(X_nodenode, Y_netnet[perm_inds, :])
            perm_cca.fit(X, Y_perm)

            perm_R = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in zip(perm_cca.x_scores_.T, perm_cca.y_scores_.T)])
            perm_Rs.append(perm_R)
        except:
            print("except")
            n_except += 1
            perm_Rs.append(np.zeros(n_keep))

    perm_Rs = np.array(perm_Rs)

    pvals = []
    for i_coef in range(n_keep):
        cur_pval = (1. + np.sum(perm_Rs[1:, 0] > actual_Rs[i_coef])) / n_permutations
        pvals.append(cur_pval)

    print("Variance explained by components:", actual_Rs)
    print("p-values:", pvals)

    pvals_loose = []
    for i_coef in range(n_keep):
        cur_pval = (1. + np.sum(perm_Rs[1:, i_coef] > actual_Rs[i_coef])) / n_permutations
        pvals_loose.append(cur_pval)
    print("p-values (loose):", pvals_loose)

    pvals = np.array(pvals)
    print('%i CCs are significant at p<0.05' % np.sum(pvals < 0.05))
    print('%i CCs are significant at p<0.01' % np.sum(pvals < 0.01))
    print('%i CCs are significant at p<0.001' % np.sum(pvals < 0.001))


def normalize_scores(Y, zscore):
    if zscore:
        return StandardScaler().fit_transform(Y)

    Y = Y.astype(float)
    for i in range(Y.shape[1]):
        max_score = np.max(Y[:, i])
        Y[:, i] = Y[:, i] / max_score

    return Y


def main():

    use_pc100 = False  # False for atlas ROI lesion load matrix
    include_cerebellum_regions = False

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/Users/hasnainmamdani/Academics/McGill/thesis/data/")
    parser.add_argument("--language_scores_only", default=True)
    args = parser.parse_args()
    print(args)

    DATA_DIR = args.data_dir

    filter_idx = get_patient_idx_with_no_previous_lesions(DATA_DIR)

    if use_pc100:
        X = get_PC_data(DATA_DIR)

    else:
        X = get_atlas_lesion_load_matrix(DATA_DIR)
        # X = np.log(1 + llm)  # for mixup after log

    X = X[filter_idx]

    if not include_cerebellum_regions:
        X = X[:, np.r_[0:111, 145:193]]

    Y = get_patient_scores(DATA_DIR, args.language_scores_only, filter_idx)

    print("X.shape", X.shape, "Y.shape", Y.shape)

    X_z = StandardScaler().fit_transform(X)
    X_deconf = deconfound(DATA_DIR, X_z)

    Y_norm = normalize_scores(Y, zscore=False)
    # Y_norm = 1 - Y_norm

    plot_cca_loadings(X_deconf, Y_norm, DATA_DIR, include_cerebellum_regions)

    permutation_test(X_deconf, Y_norm)


if __name__ == "__main__":
    main()
