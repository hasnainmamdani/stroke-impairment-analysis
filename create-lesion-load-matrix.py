from nilearn import datasets
from nilearn.image import load_img, resample_to_img
from nilearn.datasets import load_mni152_brain_mask
import os
import glob
import numpy as np
np.random.seed(39)


def get_mni_mask(reference_img):

    mask_img = load_mni152_brain_mask()
    mask_img_resampled = resample_to_img(mask_img, reference_img, interpolation="linear")
    mask = np.where(mask_img_resampled.get_fdata() == 1)  # Not using NiftiMasker because it takes too long and too much memory to transform.

    return mask


def get_binary_lesion_imgs(data_dir):

    if not os.path.isfile(data_dir + "binary_imgs.npy"):

        dataset_path = data_dir + "stroke-dataset/HallymBundang_lesionmaps_Bzdok_n1401/"
        img_filenames = glob.glob(os.path.join(dataset_path, "*.nii.gz"))
        img_filenames.sort()
        print("Number of subjects: %d" % len(img_filenames))

        mask = get_mni_mask(load_img(img_filenames[0]))

        print(mask[0].shape)
        img_data = np.empty((len(img_filenames), len(mask[0])), dtype=bool)

        for i in range(len(img_filenames)):
            print(i)
            img_data[i] = load_img(img_filenames[i]).get_fdata()[mask].astype(bool)

        np.save("binary_imgs", img_data)

    else:
        img_data = np.load(data_dir + "binary_imgs.npy")

    return img_data


def load_ho_cort_atlas(reference_img, mask):

    print("loading Harvard Oxford Cortical Atlas...")
    atlas_cort = datasets.fetch_atlas_harvard_oxford("cort-maxprob-thr50-1mm", symmetric_split=True)
    atlas_cort_img = atlas_cort.maps
    labels_cort = np.array(atlas_cort.labels)
    labels_cort = np.core.defchararray.add("HO Cortical - ", labels_cort)
    atlas_cort_img_resampled = resample_to_img(atlas_cort_img, reference_img, interpolation="nearest")  # not needed here but a good practice

    atlas_cort_vectorized = atlas_cort_img_resampled.get_fdata()[mask].astype(int)

    print(len(labels_cort))
    print(atlas_cort_img_resampled.shape)
    print(atlas_cort_vectorized.shape)
    print(np.unique(atlas_cort_vectorized))

    return atlas_cort_vectorized, labels_cort


def load_ho_subcort_atlas(reference_img, mask):

    print("loading Harvard Oxford Subcortical Atlas...")
    atlas_subcort = datasets.fetch_atlas_harvard_oxford("sub-maxprob-thr50-1mm", symmetric_split=True)
    atlas_subcort_filename = atlas_subcort.maps
    labels_subcort = np.array(atlas_subcort.labels)
    labels_subcort = np.core.defchararray.add("HO Subcortical - ", labels_subcort)
    atlas_subcort_img_resampled = resample_to_img(atlas_subcort_filename, reference_img, interpolation="nearest")  # not needed here but a good practice

    atlas_subcort_vectorized = atlas_subcort_img_resampled.get_fdata()[mask].astype(int)

    #  remove labels for cortical and white matter regions
    print("Removing following cortical and white matter regions from the subcortical atlas:", labels_subcort[1],
          labels_subcort[2], labels_subcort[13], labels_subcort[14])

    atlas_subcort_vectorized[(atlas_subcort_vectorized == 1) | (atlas_subcort_vectorized == 2) |
                             (atlas_subcort_vectorized == 13) | (atlas_subcort_vectorized == 14)] = 0

    print(len(labels_subcort))
    print(atlas_subcort_img_resampled.shape)
    print(atlas_subcort_vectorized.shape)
    print(np.unique(atlas_subcort_vectorized))

    return atlas_subcort_vectorized, labels_subcort


def load_cereb_atlas(atlas_path, reference_img, mask):

    print("loading Diedrichsen Cerebellum Atlas...")
    atlas_cereb_img = load_img(atlas_path + "Cerebellum-SUIT-FSLView/Cerebellum-SUIT-maxprob-thr25.nii")
    labels_cereb = np.loadtxt(atlas_path + "Cerebellum-SUIT-FSLView/labels.txt", dtype="U", delimiter="\n")
    labels_cereb = np.core.defchararray.add("Cerebellum - ", labels_cereb)
    atlas_cereb_img_resampled = resample_to_img(atlas_cereb_img, reference_img, interpolation="nearest")  # not needed here but a good practice

    atlas_cereb_vectorized = atlas_cereb_img_resampled.get_fdata()[mask].astype(int)

    print(len(labels_cereb))
    print(atlas_cereb_img_resampled.shape)
    print(atlas_cereb_vectorized.shape)
    print(np.unique(atlas_cereb_vectorized))

    return atlas_cereb_vectorized, labels_cereb


def load_jhu_wm_atlas(atlas_path, reference_img, mask):

    print("loading JHU White Matter Tract Atlas...")
    atlas_wm_img = load_img(atlas_path + "JHU-ICBM-labels-1mm.nii.gz")
    labels_wm = np.loadtxt(atlas_path + "labels_jhu_icbm.txt", dtype="U", delimiter="\n")
    labels_wm = np.core.defchararray.add("JHU WM - ", labels_wm)

    atlas_wm_img_resampled = resample_to_img(atlas_wm_img, reference_img, interpolation="nearest")  # not needed here but a good practice

    atlas_wm_vectorized = atlas_wm_img_resampled.get_fdata()[mask].astype(int)

    print(len(labels_wm))
    print(atlas_wm_img_resampled.shape)
    print(atlas_wm_vectorized.shape)
    print(np.unique(atlas_wm_vectorized))

    return atlas_wm_vectorized, labels_wm


def create_lesion_load_matrix_atlas(atlas, region_names, imgs):

    assert atlas.shape == imgs[0].shape, "Atlas dimesnsion and image dimension must be same"

    region_labels = np.unique(atlas)
    region_labels = region_labels[region_labels != 0]  # background

    lesion_load_matrix = np.zeros((imgs.shape[0], len(region_labels)), dtype=int)

    for i_label in range(len(region_labels)):

        idx = np.where(atlas == region_labels[i_label])[0]

        imgs_region = imgs[:, idx]

        lesion_load_matrix[:, i_label] = np.sum(imgs_region, axis=1)

    return lesion_load_matrix, region_names[region_labels]


def create_lesion_load_matrix(atlas_dir, lesion_data, reference_img):

    mask = get_mni_mask(reference_img)

    atlas_cort_img, atlas_cort_labels = load_ho_cort_atlas(reference_img, mask)
    atlas_subcort_img, atlas_subcort_labels = load_ho_subcort_atlas(reference_img, mask)
    atlas_cereb_img, atlas_cereb_labels = load_cereb_atlas(atlas_dir, reference_img, mask)
    atlas_wm_img, atlas_wm_labels = load_jhu_wm_atlas(atlas_dir, reference_img, mask)

    llm_cort, region_names_cort = create_lesion_load_matrix_atlas(atlas_cort_img, atlas_cort_labels, lesion_data)
    llm_subcort, region_names_subcort = create_lesion_load_matrix_atlas(atlas_subcort_img, atlas_subcort_labels, lesion_data)
    llm_cereb, region_names_cereb = create_lesion_load_matrix_atlas(atlas_cereb_img, atlas_cereb_labels, lesion_data)
    llm_wm, region_names_wm = create_lesion_load_matrix_atlas(atlas_wm_img, atlas_wm_labels, lesion_data)

    llm = np.concatenate([llm_cort, llm_subcort, llm_cereb, llm_wm], axis=1)
    region_names = np.concatenate([region_names_cort, region_names_subcort, region_names_cereb, region_names_wm])

    return llm, region_names


DATA_DIR = "/Users/hasnainmamdani/Academics/McGill/thesis/data/"

lesion_imgs = get_binary_lesion_imgs(DATA_DIR)
print("lesion data loaded:", lesion_imgs.shape)

reference_img = load_img("/Users/hasnainmamdani/Academics/McGill/thesis/data/stroke-dataset/HallymBundang_lesionmaps_Bzdok_n1401/1001.nii.gz")

atlas_dir = DATA_DIR + "atlas/"

lesion_load_matrix, region_labels = create_lesion_load_matrix(atlas_dir, lesion_imgs, reference_img)

print(lesion_load_matrix.shape, region_labels.shape)
np.save(DATA_DIR + "combined_lesions_load_matrix.npy", lesion_load_matrix)
np.save(DATA_DIR + "combined_atlas_region_labels.npy", region_labels)




