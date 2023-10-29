dataset=ECSSD

# python extract/extract_cosine_dot1simsiam_ds.py extract_eigs \
#     --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
#     --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
#     --which_matrix "laplacian" \
#     --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_onlysimsiam1/laplacian" \
#     --K 5

# python extract/extract_cosine_dot1simsiam_ds.py extract_single_region_segmentations \
#   --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
#   --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_onlysimsiam1/laplacian" \
#   --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation_onlysimsiam1/patches/laplacian_dino_vits16"   

#  python extract/extract_cosine_dot1simsiam_ds.py extract_crf_segmentations \
#   --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
#   --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
#   --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation_onlysimsiam1/patches/laplacian_dino_vits16" \
#   --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/single_region_segmentation_onlysimsiam1/crf/laplacian_dino_vits16" \
#   --downsample_factor 16 \
#   --num_classes 2


python extract/extract_cosine_dot1simsiam_ds.py extract_eigs \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --which_matrix "laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_cosine_
  dot1simsiam_ds_10/laplacian" \
  --K 5 \
  --epochs 10



python extract/extract_cosine_dot1simsiam_ds.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_cosine_dot1simsiam_ds_10/laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_cosine_dot1simsiam_ds_10/patches/laplacian_dino_vits16"   

python extract/extract_cosine_dot1simsiam_ds.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_cosine_dot1simsiam_ds_10/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_cosine_dot1simsiam_ds_10/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

python main.py predictions.run="srg_cosine_dot1simsiam_ds_10/crf/laplacian_dino_vits16"  