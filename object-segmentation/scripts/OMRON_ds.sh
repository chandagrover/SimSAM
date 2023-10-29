python extract/extract.py extract_features \
    --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/lists/images.txt" \
    --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/DUT-OMRON-image" \
    --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/features/dino_vits16" \
    --model_name "dino_vits16" \
    --batch_size 1

python extract/extract.py extract_eigs \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/DUT-OMRON-image" \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/features/dino_vits16" \
--which_matrix "laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/eigs_ours/laplacian" \
--K 5

python extract/extract.py extract_single_region_segmentations \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/features/dino_vits16" \
--eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/eigs_ours/laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/single_region_segmentation_ours/patches/laplacian_dino_vits16"


python extract/extract.py extract_crf_segmentations \
--images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/lists/images.txt" \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/DUT-OMRON-image" \
--segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/single_region_segmentation_ours/patches/laplacian_dino_vits16" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/single_region_segmentation_ours/crf/laplacian_dino_vits16" \
--downsample_factor 16 \
--num_classes 2
cd object-segmentation
python main.py predictions.run="DUT_OMRON/DUT-OMRON-image/single_region_segmentation_ours/crf/laplacian_dino_vits16"  

----------------------------------------------------------------------------------------------
python extract/extract.py extract_features \
    --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/lists/images.txt" \
    --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/DUT-OMRON-image" \
    --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/features/dino_vits16" \
    --model_name "dino_vits16" \
    --batch_size 1

python extract/extract.py extract_eigs \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/DUT-OMRON-image" \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/features/dino_vits16" \
--which_matrix "laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/eigs/laplacian" \
--K 5

python extract/extract.py extract_single_region_segmentations \
--features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/features/dino_vits16" \
--eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/eigs/laplacian" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/single_region_segmentation/patches/laplacian_dino_vits16"


python extract/extract.py extract_crf_segmentations \
--images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/lists/images.txt" \
--images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/DUT-OMRON-image" \
--segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/single_region_segmentation/patches/laplacian_dino_vits16" \
--output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/DUT_OMRON/DUT-OMRON-image/single_region_segmentation/crf/laplacian_dino_vits16" \
--downsample_factor 16 \
--num_classes 2

cd object-segmentation
python main.py predictions.run="DUT_OMRON/DUT-OMRON-image/single_region_segmentation/crf/laplacian_dino_vits16"  