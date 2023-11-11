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
* ECSSD
* DUTS
* DUTS-OMRON
* CUB

## **Applications**

### **Object Segmentation**
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

### **Evaluating Object Segmentation**
python eval/main.py predictions.run="srg_SimSAM/crf/laplacian_dino_vits16"
### **Semantic Segmentation**

### **Semantic CLIPStyler**







