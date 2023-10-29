python extract/extract_mmd.py extract_eigs \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
--which_matrix "laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_mmd_gaussian_sigma_zero/laplacian" \
--K 5 \
--kernel_type='gaussian' \
--sigma=0


python extract/extract_mmd.py extract_single_region_segmentations \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
--eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_mmd_gaussian_sigma_zero/laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation_mmd_gaussian_sigma_zero/patches/laplacian_dino_vits16"   

python extract/extract_mmd.py extract_crf_segmentations \
--images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
--segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation_mmd_gaussian_sigma_zero/patches/laplacian_dino_vits16" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation_mmd_gaussian_sigma_zero/crf/laplacian_dino_vits16" \
--downsample_factor 16 \
--num_classes 2
