dataset=ECSSD

python extract/extract.py extract_eigs \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
--which_matrix "laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs2/laplacian" \
--K 5
python extract/extract.py extract_single_region_segmentations \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
--eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs2/laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation2/patches/laplacian_dino_vits16"
python extract/extract.py extract_crf_segmentations \
--images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
--segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation2/patches/laplacian_dino_vits16" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation2/crf/laplacian_dino_vits16" \
--downsample_factor 16 \
--num_classes 2

python extract/extract.py extract_eigs \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
--which_matrix "laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs3/laplacian" \
--K 5
python extract/extract.py extract_single_region_segmentations \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
--eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs3/laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation3/patches/laplacian_dino_vits16"
python extract/extract.py extract_crf_segmentations \
--images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
--segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation3/patches/laplacian_dino_vits16" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation3/crf/laplacian_dino_vits16" \
--downsample_factor 16 \
--num_classes 2