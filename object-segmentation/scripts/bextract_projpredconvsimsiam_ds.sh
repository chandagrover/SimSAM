
python extract/extract.py extract_features \
    --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
    --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/arxivs/arxiv_simsiam_ds/srg_projconv2Dsimsiam_ds_10_jn/crf/laplacian_dino_vits16" \
    --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/arxivs/arxiv_simsiam_ds/features_projconv2Dsimsiam_ds_10_jn/" \
    --model_name dino_vits16 \
    --batch_size 1

python extract/extract_projconv2Dsimsiam_ds.py extract_eigs \
  --images_root "/home/chanda/Documents/Data/ECSSD/images" \
  --features_dir "/home/chanda/Documents/Data/ECSSD/features/dino_vits16" \
  --which_matrix "laplacian" \
  --output_dir "/home/chanda/Documents/Data/ECSSD/eigs_dot1projpredconv1Dsimsiam_ds_100_model_jn/laplacian" \
  --K 5 \
  --epochs 100

python extract/extract.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1PCA64linearpredlinear_dssubmax_pred2_encoder_elu_corr5" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1PCA64linearpredlinear_dssubmax_pred2_encoder_elu_corr5/patches/laplacian_dino_vits16"   

python extract/extract.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1PCA64linearpredlinear_dssubmax_pred2_encoder_elu_corr5/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1PCA64linearpredlinear_dssubmax_pred2_encoder_elu_corr5/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

cd object-segmentation

python main.py predictions.run="arxivs/arxiv_simsiam_ds/srg_dot1projconv1Dsimsiam_ds_10_jn/crf/original"

python main.py predictions.run="srg_dot1PCA64linearpredlinear_dssubmax_pred2_encoder_elu_corr5/crf/laplacian_dino_vits16"

python extract/extract.py extract_single_region_segmentations \
  --features_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/features/" \
  --eigs_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/eigs_dot1PCA64linearpredlinear_dssubmax_pred2_encoder_gelu_corr" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1PCA64linearpredlinear_dssubmax_pred2_encoder_gelu_corr/patches/laplacian_dino_vits16"   

python extract/extract.py extract_crf_segmentations \
  --images_list "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/lists/images.txt" \
  --images_root "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/images" \
  --segmentations_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1PCA64linearpredlinear_dssubmax_pred2_encoder_gelu_corr/patches/laplacian_dino_vits16" \
  --output_dir "/home/phdcs2/Hard_Disk/Datasets/Deep-Spectral-Segmentation/data/object-segmentation/ECSSD/srg_dot1PCA64linearpredlinear_dssubmax_pred2_encoder_gelu_corr/crf/laplacian_dino_vits16" \
  --downsample_factor 16 \
  --num_classes 2

cd object-segmentation

python main.py predictions.run="srg_dot1PCA64pred_dssubmax_10/crf/laplacian_dino_vits16"