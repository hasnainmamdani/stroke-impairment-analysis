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
import itertools
import seaborn as sns
from nilearn.image import load_img, resample_to_img, new_img_like, smooth_img
from nilearn.datasets import load_mni152_brain_mask
from nibabel import save
import rcca

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


def deconfound(data_dir, data):

    patient_select_df = pd.read_excel(data_dir + 'stroke-dataset/HallymBundang_MultiOutcome_25062020_selection_nopass.xlsx',
                                      sep=',', skipinitialspace=True)[:1154]

    age = StandardScaler().fit_transform(patient_select_df["Age"].values.reshape(-1, 1))
    age2 = age ** 2
    sex = pd.get_dummies(patient_select_df["Sex"]).values[:, 0].reshape(-1, 1)
    sex_x_age = sex * age
    sex_x_age2 = sex * age2
    edu = StandardScaler().fit_transform(patient_select_df["Education_years"].values.reshape(-1, 1))
    # infarct_volume = StandardScaler().fit_transform(patient_select_df["Total_infarct_volume"].values.reshape(-1, 1))
    cohort = pd.get_dummies(patient_select_df["Cohort"]).values[:, 0].reshape(-1, 1)

    # conf_mat = np.hstack([age, age2, sex, sex_x_age, sex_x_age2, edu, infarct_volume, cohort])
    conf_mat = np.hstack([age, age2, sex, sex_x_age, sex_x_age2, edu, cohort])

    data_conf = clean(data, confounds=conf_mat, detrend=False, standardize=False)

    return data_conf


def plot_cca_loadings(cca, data_dir, rcca):

    roi_names = np.load(data_dir + "combined_atlas_region_labels.npy")
    langauge_score_names = ['SVLT Immediate Recall', 'SVLT Delayed Recall', 'SVLT Recognition']

    n_comp = cca.numCC if rcca else cca.n_components
    x_loadings = cca.ws[0] if rcca else cca.x_loadings_
    y_loadings = cca.ws[1] if rcca else cca.y_loadings_

    # fixing directionality
    if not rcca:
        x_loadings[:, 0] *= -1
        x_loadings[:, 2] *= -1
        y_loadings[:, 0] *= -1
        y_loadings[:, 2] *= -1

    for i_ccomp in range(n_comp):

        plt.figure(figsize=(50, 30))
        plt.xticks(rotation=90)
        plt.tight_layout()
        # plt.ylim(-1.2, 1.2)
        # plt.yticks(np.arange(-1.2, 1.3, 0.2))
        plt.axhline(0, color="black")
        # plt.axhline(-0.6, color="red")
        # plt.axhline(0.6, color="red")
        plt.title('Canonical component %i: Atlas regions' % (i_ccomp + 1))
        plt.bar(roi_names, x_loadings[:, i_ccomp])
        plt.savefig('cca_x_%iof%i.png' % (i_ccomp + 1, n_comp), bbox_inches='tight')
        plt.savefig('cca_x_%iof%i.pdf' % (i_ccomp + 1, n_comp), bbox_inches='tight')

        plt.clf()
        plt.figure(figsize=(6, 5))
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.axhline(0, color="black")
        plt.title('Canonical component %i: Language scores' % (i_ccomp + 1))
        plt.bar(langauge_score_names, y_loadings[:, i_ccomp])
        plt.savefig('cca_y_%iof%i.png' % (i_ccomp + 1, n_comp), bbox_inches='tight')
        plt.savefig('cca_y_%iof%i.pdf' % (i_ccomp + 1, n_comp), bbox_inches='tight')


def permutation_test(actual_cca, X, Y):

    actual_Rs = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in zip(actual_cca.x_scores_.T, actual_cca.y_scores_.T)])

    n_comp = actual_cca.n_components
    n_permutations = 1000
    perm_rs = np.random.RandomState(72)
    perm_Rs = []
    n_except = 0
    for i_iter in range(n_permutations):
        if i_iter % 50 == 0:
            print(i_iter + 1)

        Y_perm = np.array([perm_rs.permutation(sub_row) for sub_row in Y])

        # same procedure, only with permuted subjects on the right side
        try:
            perm_cca = CCA(n_components=n_comp, scale=False)

            # perm_inds = np.arange(len(Y_netmet))
            # perm_rs.shuffle(perm_inds)
            # perm_cca.fit(X_nodenode, Y_netnet[perm_inds, :])
            perm_cca.fit(X, Y_perm)

            perm_R = np.array([pearsonr(X_coef, Y_coef)[0] for X_coef, Y_coef in zip(perm_cca.x_scores_.T, perm_cca.y_scores_.T)])
            perm_Rs.append(perm_R)
        except:
            print("except")
            n_except += 1
            perm_Rs.append(np.zeros(n_comp))

    perm_Rs = np.array(perm_Rs)

    pvals = []
    for i_coef in range(n_comp):
        cur_pval = (1. + np.sum(perm_Rs[1:, 0] > actual_Rs[i_coef])) / n_permutations
        pvals.append(cur_pval)

    print("Variance explained by components:", actual_Rs)
    print("p-values:", pvals)

    pvals_loose = []
    for i_coef in range(n_comp):
        cur_pval = (1. + np.sum(perm_Rs[1:, i_coef] > actual_Rs[i_coef])) / n_permutations
        pvals_loose.append(cur_pval)
    print("p-values (loose):", pvals_loose)

    pvals = np.array(pvals)
    print('%i CCs are significant at p<0.05' % np.sum(pvals < 0.05))
    print('%i CCs are significant at p<0.01' % np.sum(pvals < 0.01))
    print('%i CCs are significant at p<0.001' % np.sum(pvals < 0.001))


def plot_cca_component_comparison(cca):

    for i, j in itertools.combinations(np.arange(cca.n_components), 2):

        data_ax1 = cca.x_scores_[:, i]
        data_ax2 = cca.x_scores_[:, j]

        # fixing directionality
        if i in [0, 2]:
            data_ax1 = data_ax1 * -1
        if j in [0, 2]:
            data_ax2 = data_ax2 * -1

        gridsize = 250
        xlim = (-0.1, 0.1)
        ylim = (-0.14, 0.1)

        g = (sns.jointplot(data_ax1, data_ax2, kind="hex", xlim=xlim, ylim=ylim,
                           joint_kws=dict(gridsize=gridsize))
             .set_axis_labels('CC-'+str(i+1) + ' X', 'CC-'+str(j+1) + ' X'))
        g.ax_joint.set_xticks([-0.1, -0.05, 0, 0.05, 0.1])

        plt.subplots_adjust(left=0.15, right=0.90, top=0.95, bottom=0.10)
        g.fig.suptitle("Pearson correlation coefficient: %.3E" % pearsonr(data_ax1, data_ax2)[0])
        plt.colorbar().set_label("Number of samples")
        plt.savefig("CCA_X" + str(i+1) + str(j+1) + ".png", dpi=600)
        # plt.savefig("CCA_X" + str(i+1) + str(j+1) + ".pdf", dpi=600)

        # Cognitive scores

        data_ax1 = cca.y_scores_[:, i]
        data_ax2 = cca.y_scores_[:, j]

        # fixing directionality
        if i in [0, 2]:
            data_ax1 = data_ax1 * -1
        if j in [0, 2]:
            data_ax2 = data_ax2 * -1

        if i == 1 and j == 2:
            gridsize = 35
        else:
            gridsize = 50

        xlim = (-0.15, 0.15)
        ylim = (-0.15, 0.15) if (i == 0 and j == 1) else (-0.5, 0.5)

        g = (sns.jointplot(data_ax1, data_ax2, kind="hex", xlim=xlim, ylim=ylim,
                           joint_kws=dict(gridsize=gridsize))
             .set_axis_labels('CC-'+str(i+1) + ' Y', 'CC-'+str(j+1) + ' Y'))

        plt.subplots_adjust(left=0.15, right=0.90, top=0.95, bottom=0.10)
        g.fig.suptitle("Pearson correlation coefficient: %.3E" % pearsonr(data_ax1, data_ax2)[0])
        plt.colorbar().set_label("Number of samples")
        # plt.savefig("CCA_Y" + str(i+1) + str(j+1) + ".png", dpi=600)
        plt.savefig("CCA_Y" + str(i+1) + str(j+1) + ".pdf", dpi=600)


def normalize_scores(Y, zscore):
    if zscore:
        return StandardScaler().fit_transform(Y)

    Y = Y.astype(float)
    for i in range(Y.shape[1]):
        max_score = np.max(Y[:, i])
        Y[:, i] = Y[:, i] / max_score

    return Y


def get_mni_mask(reference_img):

    mask_img = load_mni152_brain_mask()
    mask_img_resampled = resample_to_img(mask_img, reference_img, interpolation="linear")
    mask = np.where(mask_img_resampled.get_fdata() == 1)  # Not using NiftiMasker because it takes too long and too much memory to transform.

    return mask


def plot_loadings_to_brain(data_dir, loadings, zscore=False):

    # fixing directionality. loadings.shape == (161, 3)
    loadings[:, 0] *= -1
    loadings[:, 2] *= -1

    if zscore:
        loadings = StandardScaler().fit_transform(loadings)

    reference_img = load_img(data_dir + "stroke-dataset/HallymBundang_lesionmaps_Bzdok_n1401/1001.nii.gz")
    mask_idx = get_mni_mask(reference_img)
    roi_names = np.load(data_dir + "combined_atlas_region_labels.npy")

    atlas_all_vectorized, labels_all = get_vectorized_atlas_data(data_dir)

    for i_mode in range(loadings.shape[1]):

        loadings_on_mni_vectorized = roi_to_voxels(atlas_all_vectorized, labels_all, mask_idx, roi_names,
                                                   loadings[:, i_mode])

        #  combine all atlases
        nifti_data_vectorized = np.apply_along_axis(resolve_overlap, axis=0, arr=loadings_on_mni_vectorized)

        nifti_data = np.zeros(reference_img.shape)
        nifti_data[mask_idx] = nifti_data_vectorized

        nifti_img = new_img_like(reference_img, nifti_data)
        save(nifti_img, "mode_" + str(i_mode + 1) + ("zscored" if zscore else "") + ".nii.gz")


def resolve_overlap(values):

    min_value = np.min(values)
    max_value = np.max(values)

    if max_value == 0:
        return min_value

    return max_value


def create_modewise_brain_maps(x_scores_, x_loadings_):

    recons_map = np.empty((x_loadings_.shape[1], x_scores_.shape[0], x_loadings_.shape[0]))

    for i_mode in range(x_loadings_.shape[1]):

        recons_map[i_mode] = np.matmul(x_scores_[:, i_mode].reshape(-1, 1), x_loadings_.T[i_mode].reshape(1, -1))

    return recons_map


def create_niftis_from_lesion_maps(recons_brain_map, data_dir, smooth_image=False, smooth_fwhm=10):
    # recons_brain_map.shape == (3, 1154, 161)

    reference_img = load_img(data_dir + "stroke-dataset/HallymBundang_lesionmaps_Bzdok_n1401/1001.nii.gz")
    mask_idx = get_mni_mask(reference_img)
    roi_names = np.load(data_dir + "combined_atlas_region_labels.npy")

    atlas_all_vectorized, labels_all = get_vectorized_atlas_data(data_dir)

    for i_mode in range(recons_brain_map.shape[0]):

        tt=0

        for i_patient in range(recons_brain_map.shape[1]):

            loadings_on_mni_vectorized = roi_to_voxels(atlas_all_vectorized, labels_all, mask_idx,
                                                       roi_names, recons_brain_map[i_mode][i_patient])

            #  plot back to brain space
            nifti_data_vectorized = np.apply_along_axis(resolve_overlap, axis=0, arr=loadings_on_mni_vectorized)

            nifti_data = np.zeros(reference_img.shape)
            nifti_data[mask_idx] = nifti_data_vectorized

            nifti_img = new_img_like(reference_img, nifti_data)

            if smooth_image:
                nifti_img = smooth_img(nifti_img, smooth_fwhm)

            save(nifti_img, "cca_dataset/mode_" + str(i_mode + 1) + "_patientID_" + str(i_patient) +
                 ("_smooth_" + str(smooth_fwhm) if smooth_image else "") + ".nii.gz")

            tt+=1

            if tt==3:
                break


def roi_to_voxels(atlas_all_vectorized, labels_all, mask_idx, roi_names, roi_values):

    loadings_on_mni_vectorized = np.zeros((len(atlas_all_vectorized), len(mask_idx[0])))  # 4 x 1862781 (one for each atlas)

    for i_roi in range(len(roi_names)):

        if roi_names[i_roi].startswith("HO Cortical - "):

            label_int = np.where(labels_all[0] == roi_names[i_roi])[0][0]
            idx_roi = np.where(atlas_all_vectorized[0] == label_int)
            i_atlas = 0

        elif roi_names[i_roi].startswith("HO Subcortical - "):

            label_int = np.where(labels_all[1] == roi_names[i_roi])[0][0]
            idx_roi = np.where(atlas_all_vectorized[1] == label_int)
            i_atlas = 1

        elif roi_names[i_roi].startswith("Hammer's - "):

            label_int = np.where(labels_all[2] == roi_names[i_roi])[0][0]
            idx_roi = np.where(atlas_all_vectorized[2] == label_int)
            i_atlas = 2

        elif roi_names[i_roi].startswith("JHU WM - "):

            label_int = np.where(labels_all[3] == roi_names[i_roi])[0][0]
            idx_roi = np.where(atlas_all_vectorized[3] == label_int)
            i_atlas = 3

        else:
            print("Error: unrecognizable ROI:", roi_names[i_roi])
            raise SystemExit

        loadings_on_mni_vectorized[i_atlas][idx_roi] = roi_values[i_roi]

    return loadings_on_mni_vectorized


def get_vectorized_atlas_data(data_dir):

    atlas_ho_cort_vectorized = np.load(data_dir + "atlas/processed/ho_cortical_atlas_vectorized.npy")
    atlas_ho_subcort_vectorized = np.load(data_dir + "atlas/processed/ho_subcortical_atlas_vectorized.npy")
    atlas_cereb_vectorized = np.load(data_dir + "atlas/processed/cereb_atlas_vectorized.npy")
    atlas_jhu_wm_vectorized = np.load(data_dir + "atlas/processed/jhu_wm_atlas_vectorized.npy")

    atlas_all_vectorized = [atlas_ho_cort_vectorized, atlas_ho_subcort_vectorized, atlas_cereb_vectorized, atlas_jhu_wm_vectorized]

    atlas_ho_cort_labels = np.load(data_dir + "atlas/processed/ho_cortical_atlas_region_labels.npy")
    atlas_ho_subcort_labels = np.load(data_dir + "atlas/processed/ho_subcortical_atlas_region_labels.npy")
    atlas_cereb_labels = np.load(data_dir + "atlas/processed/cereb_atlas_region_labels.npy")
    atlas_jhu_wm_labels = np.load(data_dir + "atlas/processed/jhu_wm_atlas_region_labels.npy")

    labels_all = [atlas_ho_cort_labels, atlas_ho_subcort_labels, atlas_cereb_labels, atlas_jhu_wm_labels]

    return atlas_all_vectorized, labels_all


def main():

    use_pc100 = False  # False for atlas ROI lesion load matrix

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

    Y = get_patient_scores(DATA_DIR, args.language_scores_only, filter_idx)

    print("X.shape", X.shape, "Y.shape", Y.shape)

    X_z = StandardScaler().fit_transform(X)
    X_deconf = deconfound(DATA_DIR, X_z)

    Y_norm = normalize_scores(Y, zscore=False)
    Y_norm = 1 - Y_norm

    # Y_deconf = deconfound(DATA_DIR, Y_norm)

    r_cca = False

    if r_cca:
        # X_deconf -= X_deconf.mean(axis=0)
        # Y_norm -= Y_norm.mean(axis=0)

        cca = rcca.CCA(kernelcca=False, numCC=3).train([X_deconf, Y_norm])
    else:
        cca = CCA(n_components=3, scale=False).fit(X_deconf, Y_norm)

    # plot_cca_loadings(cca, DATA_DIR, rcca=r_cca)

    # plot_cca_component_comparison(cca)

    # plot_loadings_to_brain(DATA_DIR, cca.x_loadings_, zscore=True)

    # permutation_test(cca, X_deconf, Y_norm)

    recons_brain_map = create_modewise_brain_maps(cca.x_scores_, cca.x_loadings_)  # (3, 1154, 161)
    create_niftis_from_lesion_maps(recons_brain_map, DATA_DIR, smooth_image=True, smooth_fwhm=5)


if __name__ == "__main__":
    main()
