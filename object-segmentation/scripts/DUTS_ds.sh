python extract/extract.py extract_features \
    --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/lists/images.txt" \
    --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/DUTS-TR-Image" \
    --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/DUTS-TR-Image/features/dino_vits16" \
    --model_name "dino_vits16" \
    --batch_size 1

python extract/extract.py extract_eigs \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/DUTS-TR-Image" \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/DUTS-TR-Image/features/dino_vits16" \
--which_matrix "laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/DUTS-TR-Image/eigs/laplacian" \
--K 5

python extract/extract.py extract_single_region_segmentations \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/features/dino_vits16" \
--eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/eigs_ours/laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/single_region_segmentation_ours/patches/laplacian_dino_vits16"


python extract/extract.py extract_crf_segmentations \
--images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/lists/images.txt" \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/DUTS-TR-Image" \
--segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/single_region_segmentation_ours/patches/laplacian_dino_vits16" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUTS/DUTS-TR/single_region_segmentation_ours/crf/laplacian_dino_vits16" \
--downsample_factor 16 \
--num_classes 2
cd object-segmentation
python main.py predictions.run="DUTS-TR/single_region_segmentation/crf/laplacian_dino_vits16"  