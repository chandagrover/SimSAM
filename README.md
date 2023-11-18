# SimSAM
This code accompanies the paper "**Simple Siamese Representations based Semantic Affinity Matrix with Applications in Segmentation and Style Transfer**"

## **Abstract**
Deep spectral methods have shown significant performance gains in image segmentation under unsupervised settings. Segmentation masks are obtained via spectral segmentation performed using the Laplacian of the feature affinity matrix (computed on pre-trained DINO features). As such, the computation of these feature affinity matrices is essential for segmentation. We introduce a novel framework SimSAM, based on a Simple Siamese network, to compute a semantically consistent, dense feature affinity matrix for spectral segmentation. We train the projector and predictor of the Siamese Neural Network in a non-contrastive way to compute the correlations of the dense features extracted from DINO-ViT. Our experimental results show improvements in object segmentation and text-based image stylization.
## **Examples**
## **How to Run**
### **Dependencies**
The dependencies are listed in the requirements.txt
### **Data Preparation**
- [ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)
- [DUTS](http://saliencydetection.net/duts/)
-  [DUTS-OMRON](http://saliencydetection.net/dut-omron/)
- [CUB](https://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
- [Pascal VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012//)

## **Algorithm: SimSAM**
#### **Step 1: Feature Extraction**
`python extract/extract_SimSAM.py extract_features \
    --images_list "/path/to/lists/images.txt" \    
    --images_root "/path/to/dataset_name/images/" \    
    --output_dir "/path/to/dataset_name/features/" \    
    --model_name dino_vits16 \    
    --batch_size 1`
#### **Step 2: Eigen Vector Computation with Semantic Affinity Matrix**
`python extract/extract_SimSAM.py extract_eigs \
--images_root "/path/to/dataset_name/images/" \
--features_dir "/path/to/dataset_name/features/" \
--which_matrix "laplacian" \
--output_dir "/path/to/dataset_name/eigs_SimSAM/laplacian" \
--K 15`
#### **Or Pre-Trained EigenVectors computed with SimSAM on the below datasets**
* [ECSSD](https://drive.google.com/drive/folders/1-16ckaN8wuBu04kl198zvstpTXCC0Lix)
* [DUTS](https://drive.google.com/drive/folders/1u4GmeteUWb1G-R25OhZqSK_B1PswrSNF)
* [DUTS-OMRON](https://drive.google.com/drive/folders/1d2p20ZPQYFKDxioFFnJnrESqeIuPA-Rw)
* [CUB](https://drive.google.com/drive/folders/1xuf5Qs1y8p7Pg6iFwW5S0P4smDp4vOdu)

## **Applications**

### **1. Object Segmentation**
`cd object-segmentation`

`python extract/extract_SimSAM.py extract_single_region_segmentations \
--features_dir "/path/to/dataset_name/features/" \
--eigs_dir "/path/to/dataset_name/eigs_SimSAM/laplacian" \
--output_dir "path/to/dataset_name/srg_SimSAM/patches/laplacian_dino_vits16"  `

`python extract/extract_SimSAM.py extract_crf_segmentations \
--images_list "/path/to/lists/images.txt" \ 
--images_root "/path/to/dataset_name/images/" \
--segmentations_dir "path/to/dataset_name/srg_SimSAM/patches/laplacian_dino_vits16" \
--output_dir"path/to/dataset_name/srg_SimSAM/srg_SimSAM/crf/laplacian_dino_vits16" \
--downsample_factor 16 \
--num_classes 2`

#### **Evaluating Object Segmentation**
`python eval/main.py predictions.run="srg_SimSAM/crf/laplacian_dino_vits16"`
### **2. Semantic Segmentation**
##### Creating Multi Region Segmementation
`python extract/extract_SimSAM.py extract_multi_region_segmentations \
    --non_adaptive_num_segments 15 \
    --features_dir "/path/to/dataset_name/features/dino_vits16" \
    --eigs_dir "/path/to/dataset_name/eigs/laplacian_dino_vits16" \
    --output_dir "/path/to/dataset_name/srg/multi_region_segmentation/laplacian_dino_vits16"`

###### Extract bounding boxes
`python extract/extract_SimSAM.py extract_bboxes \
    --features_dir "/path/to/dataset_name/features/dino_vits16" \
    --segmentations_dir "/path/to/dataset_name/srg/multi_region_segmentation/laplacian_dino_vits16" \
    --num_dilate 5 \
    --downsample_factor 16 \
    --output_file "/path/to/dataset_name/srg/multi_region_bboxes/laplacian_dino_vits16/bboxes.pth"`

###### Extract bounding box features
`python extract/extract_SimSAM.py extract_bbox_features \
    --model_name "dino_vits16" \
    --images_root "/path/to/dataset_name/images/trainval/JPEGImages" \
    --bbox_file "/path/to/dataset_name/srg/multi_region_bboxes/laplacian_dino_vits16/bboxes.pth" \
    --output_file "/path/to/dataset_name/srg/multi_region_bboxes/laplacian_dino_vits16/bbox_features.pth"`

###### Extract clusters
`python extract/extract_SimSAM.py extract_bbox_clusters \
    --bbox_features_file "/path/to/dataset_name/srg/multi_region_bboxes/laplacian_dino_vits16/bbox_features.pth" \
    --output_file "/path/to/dataset_name/srg/multi_region_bboxes/laplacian_dino_vits16/bbox_clusters.pth"` 

###### Create semantic segmentations
`python extract/extract.py extract_semantic_segmentations \
    --segmentations_dir "/path/to/dataset_name/srg/multi_region_segmentation/laplacian_dino_vits16" \
    --bbox_clusters_file "/path/to/dataset_name/srg/multi_region_bboxes/laplacian_dino_vits16/bbox_clusters.pth" \
    --output_dir "/path/to/dataset_name/srg/semantic_segmentations/patches/laplacian_dino_vits16/segmaps"`

##### Visualize Semantic Segmetnation Outputs
`streamlit run extract/extract.py vis_segmentations -- \
--images_list "/path/to/dataset_name/lists/images.txt" \
--images_root "/path/to/dataset_name/images/trainval/JPEGImages" \
--segmentations_dir "/path/to/dataset_name/srg/semantic_segmentations/patches/laplacian_dino_vits16/segmaps" \
--bbox_file "/path/to/dataset_name/srg/multi_region_bboxes/laplacian_dino_vits16/bboxes.pth"`
### **3. Semantic CLIPStyler**
To stylize the segmented content image with text description.

`    python clipstyler_spectral.py --content_path "/path/to/content_image.png" --segmentedImage_path "/path/to/segmented_file.npy" --filename "filename" --exp_name "exp1" --text "Starry Night by Vincent van gogh".`







