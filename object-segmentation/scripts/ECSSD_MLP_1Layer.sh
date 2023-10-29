
python extract/extract_PCA_mlp_1Layer.py extract_eigs \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
--which_matrix "laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_PCA128_mlp1Layer/laplacian" \
--K 5


python extract/extract.py extract_single_region_segmentations \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features" \
--eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/arxivs/arxiv_PCA_pred_siam/eigs_dot1PCA32pred_ds_10" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/arxivs/arxiv_PCA_pred_siam/srg_dot1PCA32pred_ds_10/patches/laplacian_dino_vits16"   

python extract/extract.py extract_crf_segmentations \
--images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
--segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1PCA32pred_ds_10/patches/laplacian_dino_vits16" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1PCA32pred_ds_10/crf/laplacian_dino_vits16" \
--downsample_factor 16 \
--num_classes 2

cd object-segmentation

python main.py predictions.run="srg_PCA128_mlp1Layer/crf/laplacian_dino_vits16"


--------------------------------------------------------------------------------------
python extract/extract_mlp_1Layer.py extract_eigs \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/" \
--which_matrix "laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_mlp_1Layer/laplacian" \
--K 5

python extract/extract.py extract_single_region_segmentations \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features" \
--eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_mlp_1Layer/laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation_mlp_1Layer/patches/laplacian_dino_vits16"   

python extract/extract.py extract_crf_segmentations \
--images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
--segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation_mlp_1Layer/patches/laplacian_dino_vits16" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation_mlp_1Layer/crf/laplacian_dino_vits16" \
--downsample_factor 16 \
--num_classes 2

python main.py predictions.run="single_region_segmentation_mlp_1Layer/crf/laplacian_dino_vits16"