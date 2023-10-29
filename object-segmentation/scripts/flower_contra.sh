dataset=Flowers/102flowers
python extract/extract_contra.py extract_eigs \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/$dataset/jpg" \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/$dataset/features/dino_vits16" \
--which_matrix "laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/$dataset/eigs_contrastive/laplacian" \
--K 5


python extract/extract_contra.py extract_single_region_segmentations \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/$dataset/features/dino_vits16" \
--eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/$dataset/eigs_contrastive/laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/$dataset/single_region_segmentation_contra/patches/laplacian_dino_vits16"   

python extract/extract_contra.py extract_crf_segmentations \
--images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/$dataset/lists/images.txt" \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/$dataset/jpg" \
--segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/$dataset/single_region_segmentation_contra/patches/laplacian_dino_vits16" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/$dataset/single_region_segmentation_contra/crf/laplacian_dino_vits16" \
--downsample_factor 16 \
--num_classes 2