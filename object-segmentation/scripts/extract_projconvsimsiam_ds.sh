python extract/extract_projconv2Dsimsiam_ds.py extract_eigs \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --which_matrix "laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_projconvsimsiam_ds_10/laplacian" \
  --K 5 \
  --epochs 10


python extract/extract_projconv2Dsimsiam_ds.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_projconvsimsiam_ds_10/laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_projconvsimsiam_ds_10/patches/laplacian_dino_vits16"   

python extract/extract_projconv2Dsimsiam_ds.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_projconvsimsiam_ds_10/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_projconvsimsiam_ds_10/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

python main.py predictions.run="srg_projconvsimsiam_ds_10/crf/laplacian_dino_vits16"

------------------------------------------------------------------------------------------

python extract/extract_projconv2Dsimsiam_ds.py extract_eigs \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --which_matrix "laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1projconv1Dsimsiam_ds_10_jn/laplacian" \
  --K 5 \
  --epochs 100


python extract/extract_dot1projconv2Dsimsiam_ds.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1projconv1Dsimsiam_ds_10_jn" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projconv1Dsimsiam_ds_10_jn/patches/laplacian_dino_vits16"   

python extract/extract_dot1projconv2Dsimsiam_ds.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projconv1Dsimsiam_ds_10_jn/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projconv1Dsimsiam_ds_10_jn/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

python main.py predictions.run="srg_dot1projconv1Dsimsiam_ds_10_jn/crf/laplacian_dino_vits16"

----------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------

python extract/extract_projconv2Dsimsiam_ds.py extract_eigs \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --which_matrix "laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1projpredconv1Dsimsiam_ds_10_jn/laplacian" \
  --K 5 \
  --epochs 10


python extract/extract_projconv2Dsimsiam_ds.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1projpredconv1Dsimsiam_ds_10_jn" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1Dsimsiam_ds_10_jn/patches/laplacian_dino_vits16"   

python extract/extract_projconv2Dsimsiam_ds.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1Dsimsiam_ds_10_jn/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1Dsimsiam_ds_10_jn/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

python main.py predictions.run="srg_dot1projpredconv1Dsimsiam_ds_10_jn/crf/laplacian_dino_vits16"
--------------------------------------------------------------------------------------------

python extract/extract_projconv2Dsimsiam_ds.py extract_eigs \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --which_matrix "laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1projpredconv1Dsimsiam_dot9ds_10_jn/laplacian" \
  --K 5 \
  --epochs 10


python extract/extract_projconv2Dsimsiam_ds.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1projpredconv1Dsimsiam_dot9ds_10_jn" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1Dsimsiam_dot9ds_10_jn/patches/laplacian_dino_vits16"   

python extract/extract_projconv2Dsimsiam_ds.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1Dsimsiam_dot9ds_10_jn/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1Dsimsiam_dot9ds_10_jn/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

python main.py predictions.run="srg_dot1projpredconv1Dsimsiam_dot9ds_10_jn/crf/laplacian_dino_vits16"


--------------------------------------------------------------------------------------------

python extract/extract_projconv2Dsimsiam_ds.py extract_eigs \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --which_matrix "laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot3projpredconv1Dsimsiam_dot7ds_10_jn/laplacian" \
  --K 5 \
  --epochs 10


python extract/extract_projconv2Dsimsiam_ds.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot3projpredconv1Dsimsiam_dot7ds_10_jn" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot3projpredconv1Dsimsiam_dot7ds_10_jn/patches/laplacian_dino_vits16"   

python extract/extract_projconv2Dsimsiam_ds.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot3projpredconv1Dsimsiam_dot7ds_10_jn/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot3projpredconv1Dsimsiam_dot7ds_10_jn/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

python main.py predictions.run="srg_dot3projpredconv1Dsimsiam_dot7ds_10_jn/crf/laplacian_dino_vits16"

---------------------------------------------------------------------------------------------

python extract/extract_projconv2Dsimsiam_ds.py extract_eigs \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --which_matrix "laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1projpredconv1DGBsimsiam_dot9ds_10_jn/laplacian" \
  --K 5 \
  --epochs 10


python extract/extract_projconv2Dsimsiam_ds.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1projpredconv1DGBsimsiam_dot9ds_10_jn" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1DGBsimsiam_dot9ds_10_jn/patches/laplacian_dino_vits16"   

python extract/extract_projconv2Dsimsiam_ds.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1DGBsimsiam_dot9ds_10_jn/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1DGBsimsiam_dot9ds_10_jn/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

python main.py predictions.run="srg_dot1projpredconv1DGBsimsiam_dot9ds_10_jn/crf/laplacian_dino_vits16"

------------------------------------------------------------------------------------------

python extract/extract_projconv2Dsimsiam_ds.py extract_eigs \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --which_matrix "laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1projpredconv1Dsimsiam_dot9ds_100_jn/laplacian" \
  --K 5 \
  --epochs 100


python extract/extract_projconv2Dsimsiam_ds.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1projpredconv1Dsimsiam_dot9ds_100_jn" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1Dsimsiam_dot9ds_100_jn/patches/laplacian_dino_vits16"   

python extract/extract_projconv2Dsimsiam_ds.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1Dsimsiam_dot9ds_100_jn/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1Dsimsiam_dot9ds_100_jn/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

python main.py predictions.run="srg_dot1projpredconv1Dsimsiam_dot9ds_100_jn/crf/laplacian_dino_vits16"

-----------------------------------------------------------------------------------



---------------------------------------------------------------------------------------
python extract/extract_projconv2Dsimsiam_ds.py extract_eigs \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --which_matrix "laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1projpredconv1Dsimsiam_dot9ds_100_jn/laplacian" \
  --K 5 \
  --epochs 100


python extract/extract_projconv2Dsimsiam_ds.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1projpredconv1Dsimsiam_ds_10best3_model_jn" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1Dsimsiam_ds_10best3_model_jn/patches/laplacian_dino_vits16"   

python extract/extract_projconv2Dsimsiam_ds.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1Dsimsiam_ds_10best3_model_jn/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1projpredconv1Dsimsiam_ds_10best3_model_jn/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

python main.py predictions.run="srg_dot1projpredconv1Dsimsiam_ds_10best3_model_jn/crf/laplacian_dino_vits16"

---------------------------------------------------------------------------------

python extract/extract_projconv2Dsimsiam_ds.py extract_eigs \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --which_matrix "laplacian" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1PCANetpredconv1Dsimsiam_ds_10_model_jn/laplacian" \
  --K 5 \
  --epochs 100


python extract/extract_projconv2Dsimsiam_ds.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/dino_vits16" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1PCANetpredconv1Dsimsiam_ds_10_model_jn" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1PCANetpredconv1Dsimsiam_ds_10_model_jn/patches/laplacian_dino_vits16"   

python extract/extract_projconv2Dsimsiam_ds.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1PCANetpredconv1Dsimsiam_ds_10_model_jn/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1PCANetpredconv1Dsimsiam_ds_10_model_jn/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2
cd object-segmentation
python main.py predictions.run="srg_dot1PCANetpredconv1Dsimsiam_ds_10_model_jn/crf/laplacian_dino_vits16"