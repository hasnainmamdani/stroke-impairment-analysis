from nibabel import save
from nilearn.image import load_img, resample_to_img, new_img_like
from nilearn.datasets import load_mni152_brain_mask
import numpy as np
from nilearn import plotting
np.random.seed(39)


def get_mni_mask(reference_img):

    mask_img = load_mni152_brain_mask()
    mask_img_resampled = resample_to_img(mask_img, reference_img, interpolation="linear")
    mask = np.where(mask_img_resampled.get_fdata() == 1)  # Not using NiftiMasker because it takes too long and too much memory to transform.

    return mask


def get_vectorized_atlas_data(data_dir):

    atlas_dir = data_dir + "atlas/processed/"

    ho_cortical_atlas_vectorized = np.load(atlas_dir + "ho_cortical_atlas_vectorized.npy")
    ho_subcortical_atlas_vectorized = np.load(atlas_dir + "ho_subcortical_atlas_vectorized.npy")
    cereb_atlas_vectorized = np.load(atlas_dir + "cereb_atlas_vectorized.npy")
    jhu_wm_atlas_vectorized = np.load(atlas_dir + "jhu_wm_atlas_vectorized.npy")

    atlas_all_vectorized = [ho_cortical_atlas_vectorized, ho_subcortical_atlas_vectorized, cereb_atlas_vectorized,
                            jhu_wm_atlas_vectorized]

    ho_cortical_atlas_region_labels = np.load(atlas_dir + "ho_cortical_atlas_region_labels.npy")
    ho_subcortical_atlas_region_labels = np.load(atlas_dir + "ho_subcortical_atlas_region_labels.npy")
    cereb_atlas_region_labels = np.load(atlas_dir + "cereb_atlas_region_labels.npy")
    jhu_wm_atlas_region_labels = np.load(atlas_dir + "jhu_wm_atlas_region_labels.npy")

    labels_all = [ho_cortical_atlas_region_labels, ho_subcortical_atlas_region_labels, cereb_atlas_region_labels,
                  jhu_wm_atlas_region_labels]

    return atlas_all_vectorized, labels_all


def roi_to_voxels(atlas_all_vectorized, labels_all, roi_names, roi_values):

    loadings_on_mni_vectorized = np.zeros(
        (len(atlas_all_vectorized), len(atlas_all_vectorized[0])))  # 4 x 1862781 (one for each atlas)

    for i_roi in range(len(roi_names)):

        if roi_names[i_roi].startswith("HO Cortical - "):

            i_atlas = 0
            label_int = np.where(labels_all[i_atlas] == roi_names[i_roi])[0][0]
            idx_roi = np.where(atlas_all_vectorized[i_atlas] == label_int)

        elif roi_names[i_roi].startswith("HO Subcortical -"):

            i_atlas = 1
            label_int = np.where(labels_all[i_atlas] == roi_names[i_roi])[0][0]
            idx_roi = np.where(atlas_all_vectorized[i_atlas] == label_int)

        elif roi_names[i_roi].startswith("Cerebellum - "):

            i_atlas = 2
            label_int = np.where(labels_all[i_atlas] == roi_names[i_roi])[0][0]
            idx_roi = np.where(atlas_all_vectorized[i_atlas] == label_int)

        else:  # white matter atlas

            i_atlas = 3
            label_int = np.where(labels_all[i_atlas] == roi_names[i_roi])[0][0]
            idx_roi = np.where(atlas_all_vectorized[i_atlas] == label_int)

        loadings_on_mni_vectorized[i_atlas][idx_roi] = roi_values[i_roi]

    return loadings_on_mni_vectorized


def resolve_overlap(values):
    min_value = np.min(values)
    max_value = np.max(values)

    if max_value == 0:
        return min_value

    return max_value


def plot_lesion_volumes_summary_to_nifti(data_dir, atlas_labels_filename, lesion_volumes_avg, reference_img,
                                         plot_output_file):

    atlas_all_vectorized, labels_all = get_vectorized_atlas_data(data_dir)

    roi_names = np.load(data_dir + atlas_labels_filename)
    mask_idx = get_mni_mask(reference_img)

    loadings_on_mni_vectorized = roi_to_voxels(atlas_all_vectorized, labels_all, roi_names, lesion_volumes_avg)

    #  combine all atlases
    nifti_data_vectorized = np.apply_along_axis(resolve_overlap, axis=0, arr=loadings_on_mni_vectorized)

    nifti_data = np.zeros(reference_img.shape)
    nifti_data[mask_idx] = nifti_data_vectorized

    nifti_img = new_img_like(reference_img, nifti_data)
    save(nifti_img, plot_output_file + ".nii.gz")

    display = plotting.plot_roi(nifti_img, colorbar=True, cmap='YlOrRd', display_mode='z', cut_coords=7, vmin=0.03,
                           bg_img=data_dir + 'atlas/colin.nii.gz', black_bg=True)
    display.annotate(size=15)
    display.savefig(plot_output_file, dpi=1024)


def main():

    DATA_DIR = "/Users/hasnainmamdani/Academics/McGill/thesis/data/"

    reference_img = load_img(DATA_DIR + "stroke-dataset/HallymBundang_lesionmaps_Bzdok_n1401_old/1001.nii.gz")

    lesion_load_matrix = np.load(DATA_DIR + "llm/combined_lesions_load_matrix.npy")
    region_labels = np.load(DATA_DIR + "llm/combined_lesions_load_matrix.npy")

    print(lesion_load_matrix.shape, region_labels.shape)

    atlas_labels_filename = "llm/combined_atlas_region_labels.npy"

    lesion_volumes_avg = np.sum(lesion_load_matrix, axis=0) / lesion_load_matrix.shape[0]

    plot_lesion_volumes_summary_to_nifti(DATA_DIR, atlas_labels_filename, lesion_volumes_avg, reference_img,
                                         DATA_DIR + "lesion_volumes_summary")

    # lesion_volumes_log_avg = np.sum(np.log(1 + lesion_load_matrix), axis=0) / lesion_load_matrix.shape[0]
    #
    # plot_lesion_volumes_summary_to_nifti(DATA_DIR, atlas_labels_filepath, lesion_volumes_log_avg, reference_img,
    #                                      DATA_DIR + "lesion_volumes_log_summary")


if __name__ == "__main__":
    main()
