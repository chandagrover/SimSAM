dataset=ECSSD

python extract/extract_only_simsiam.py extract_eigs \
    --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD_5/images" \
    --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD_5/features/dino_vits16" \
    --which_matrix "laplacian" \
    --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD_5/eigs_onlysimsiam_py/laplacian" \
    --K 5

python extract/extract_only_simsiam.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD_5/features/dino_vits16" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD_5/eigs_onlysimsiam_py/laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD_5/srg_onlysimsiam_py/patches/laplacian_dino_vits16" 

 python extract/extract_only_simsiam.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD_5/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD_5/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD_5/srg_onlysimsiam_py/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD_5/srg_onlysimsiam_py/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

-------------------------------------------------------------
python extract/extract_only_simsiam.py extract_eigs \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --which_matrix "laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_onlysimsiam2_200/laplacian" \
  --K 5

python extract/extract_only_simsiam.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_onlysimsiam2_200" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_onlysimsiam2_200/patches/laplacian_dino_vits16"   

python extract/extract_only_simsiam.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_onlysimsiam2_200/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_onlysimsiam2_200/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

python main.py predictions.run="srg_onlysimsiam2_200/crf/laplacian_dino_vits16"

--------------------------------------------------------------

