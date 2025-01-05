# fathomnet-out-of-sample-detection
This directory contains the codebase, models and data to run object detection and out-of-sample prediction for the dataset provided by the [Kaggle Dataset](https://www.kaggle.com/competitions/fathomnet-out-of-sample-detection/overview) for FathomNet 2023. The dataset was curated by a single camera system deployed by the Monterey Bay Aquarium Research Institute (MBARI) on several of its Remotely Operated Vehicles off the coast of Central California. The `train.json` file was generously labeled by the incredibly talented researched at MBARI.


## Important Note:
The dataset files are not on GitHub. The data files are accessible at this Google Drive link: [https://drive.google.com/drive/u/0/folders/1vxP-u9smnqU8ffGsXKKl1qkdWixVDVcY](https://drive.google.com/drive/u/0/folders/1vxP-u9smnqU8ffGsXKKl1qkdWixVDVcY). The below directory structure described includes the data directory descriptions.

## Directory Structure

### multilabel_classification/
- `data.csv`
    - Provided in the Kaggle dataset:
    - Each line of the csv files indicates an image by its id and a list of corresponding categories present in the frame.
    ```
    id, categories
    4a7f2199-772d-486d-b8e2-b651246316b5, [1.0]
    3bddedf6-4ff8-4e81-876a-564d2b03b364, "[1.0, 9.0, 11.0, 88.0]"
    3f735021-f5de-4168-b139-74bf2859d12a, "[1.0, 37.0, 51.0, 119.0]"
    130e185f-09c5-490c-8d08-641c4cbf6e54, "[1.0, 51.0, 119.0]"
    ```
    - The ids correspond those in the object detection files. The categories are the set of all unique annotations found in the associated image.

### object_detection/
The datasets are formatted to adhere to the COCO Object Detection standard. Every training image contains at least one annotation corresponding to a category_id ranging from 1 to 290 and a supercategory from 1 to 20. The fine-grained annotations are taxonomic thought not always at the same level of the taxonomic tree.

The training and test images for the competition were all collected in the Monterey Bay Area between the surface and 1300 meters depth. The images contain bounding box annotations of 290 categories of bottom dwelling animals. The training and evaluation data are split across an 800 meter depth threshold: all training data is collected from 0-800 meters, evaluation data comes from the whole 0-1300 meter range. Since an organisms' habitat range is partially a function of depth, the species distributions in the two regions are overlapping but not identical. Test images are drawn from the same region but may come from above or below the depth horizon.
- `train.json`
    - This is labeled data collected from 0-800m depth.
- `eval.json`
    - This is unlabeled data collected from 0-1300m depth.

### category_key.csv
- This is the csv file containing the mapping for a category id and the corresponding species name and the supercategory.

### 60-40-dataset.yaml, dataset.yaml, preprocessed-dataset.yaml
These are customized specifications for the YOLO models. Each differ in the dataset used to train the model.
- `60-40-dataset.yaml` specifies to use the 60/40 training split dataset specified at `train2/images` and `val2/images`.
- `dataset.yaml` specified to use the 80/20 training split dataset specified at `train/images` and `val/images`.
- `preprocessed-dataset.yaml` specified to use the 60/40 training split preprocessed dataset specified at `preprocessed/train/images` and `preprocessed/val/images`.

### train/
This is the folder containing the images and labels for the training set from an 80/20 split.
- `images/`: The corresponding images.
- `labels/`: The corresponding labels.
- `labels.cache`: Used to speed up the initial lookup in YOLO model training. Needs to be deleted if training different models.

### val/
This is the folder containing the images and labels for the validation set from an 80/20 split.
- `images/`: The corresponding images.
- `labels/`: The corresponding labels.
- `labels.cache`: Used to speed up the initial lookup in YOLO model training. Needs to be deleted if training different models.

### val1/
This is the folder containing the images from the eval.json file.
- *.png: The images from eval.json

### train2/
This is the folder containing the images and labels for the training set from an 60/40 split.
- `images/`: The corresponding images.
- `labels/`: The corresponding labels.
- `labels.cache`: Used to speed up the initial lookup in YOLO model training. Needs to be deleted if training different models.

### val2/
This is the folder containing the images and labels for the validation set from an 60/40 split.
- `images/`: The corresponding images.
- `labels/`: The corresponding labels.
- `labels.cache`: Used to speed up the initial lookup in YOLO model training. Needs to be deleted if training different models.

### preprocessed/
This is the folder containing the images and labels from running the Wavelet Dual-Stream Network algorithm on the images for a 60/40 split.
- `original`: The original 5950 images.
- `train/`: The corresponding training images and labels.
    - `images/`: The corresponding images.
    - `labels/`: The corresponding labels.
    - `labels.cache`: Used to speed up the initial lookup in YOLO model training. Needs to be deleted if training different models.
- `val/`: The corresponding validation images and labels.
    - `images/`: The corresponding images.
    - `labels/`: The corresponding labels.
    - `labels.cache`: Used to speed up the initial lookup in YOLO model training. Needs to be deleted if training different models.

### resampled/
This is the folder containing the images and labels from running doing resampling on the dataset by filtering all images that do not contain any of the majority 50 classes on the images for an 80/20 split.
- `images/`: The corresponding images.
    - `train/`: The corresponding training images.
    - `val/`: The corresponding validation images.
- `labels/`: The corresponding labels.
    - `train/`: The corresponding training labels.
    - `val/`: The corresponding validation labels.
- `top50-train.json`: The newly filtered json file corresponding to the annotations and images from the resampled dataset.
- `top50-train-renumbered.json`: The `top50-train.json` with the annotations renumbered (due to deleting some annotations and images).
- `train_split.json`: The corresponding training split json file required for training Fast RCNN.
- `val_split.json`: The corresponding validation split json file required for training Fast RCNN.

### training_outputs/
This is the folder containing all of the model training and evaluation results.
- `yolov8m_baseline/`: This contains the model training and evaluation results from the baseline model YOLOv8m on 200 epochs for a 60/40 split.
- `yolov8m_preprocessed/`: This contains the model training and evaluation results from the YOLOv8m on 200 epochs for a 60/40 split run on the preprocessed dataset.
- `yolov8m_finetuned`: This contains the model training and evaluation results from the YOLOv8m on 200 epochs for a 60/40 split run using the weights learned from the `yolov8m_preprocessed` model.
- `yolov8m-majority`: This contains the model training and evaluation results from the resampled data from the YOLOv8 on 200 epochs for an 80/20 split.
- `yolov8m_resampled`: This contains the model training and evaluation results from the YOLOv8m on 200 epochs for an 80/20 split.
- `yolov11m_baseline`: This contains the model training and evaluation results from the YOLOv11m on 200 epochs for a 60/40 split.
- `yolo11l_resampled`: This contains the model training and evaluation results from the YOLOv11L on 200 epochs for an 80/20 split.

### codebase/
This is the folder containing all of the Colab Notebooks used to train the models, download / process the data and do out-of-sample detection. We have multiple notebooks, as we executed it in parallel to reduce, so we did not need to wait for other notebooks to finish execution.
- `download_data.ipynb`: Preprocesses the images by applying the Wavelet Dual Stream Network algorithm.
- `fastrcnn.ipynb`: Attempts to run the Fast R-CNN on the dataset.
- `FathomNet 2023 - Baseline YOLOv8`:
    - This is the MAIN notebook. 
    - It downloads the images and annotations
    - Performs the analysis on the dataset
    - Establishes the baseline YOLOv8m model.
    - Trains YOLOv8m on the Preprocessed dataset, and attempts to fine-tune it.
- `YOLOv11-fathomnet.ipynb`:
    - A secondary notebook, it creates a proper 80/20 split by downloading the corresponding images and annotations with a split.
    - Trains the dataset on the YOLOv8m with the 80/20 split.
    - Trains the dataset on the YOLOv11L with the 80/20 split.
- `create-60-40-yolov11.ipynb`:
    - Creates a 60/40 split by downloading the corresponding images and annotations with a split.
    - Trains the dataset on the YOLOv11m with the 60/40 split. 

