This project aims to train a YOLOv8 model for object detection specifically on toothbrushes, scissors, and computer mice using a subset of the COCO 2017 dataset. 

How To Run: Simply connect to a Google Colab Runtime (Preferably L4 or A100) and run all cells. The initial cells will make sure to install any libraries necessary for the ML frameworks,
subsequent cells will download the COCO dataset. From there, the dataset is parsed for relevant images of toothbrushes, computer mice, and scissors for training and testing. 
If you want to load the trained model's weights, unzip the model and load it using the below command: 
/*
import zipfile
import os
# Unzip the model
with zipfile.ZipFile('/path/to/your/yolov8_coco.zip', 'r') as zip_ref:
    zip_ref.extractall('/content/yolov8_coco')
# Load the model
from ultralytics import YOLO
model = YOLO('/content/yolov8_coco/best.pt')
*/
From there, run the visualization code in the bottom cell with the initialized model. To 
input different images, simply upload them to the local Colab environment and change the paths in Image_paths to those that you 
desire to run the model on. 

The model was trained through a YOLOv8 framework, with object detection treates as a single regression problem. 
The image is uniquely divided into a grid cell system with each grid cell responsible for predicting
bounding boxes and confidence scores. This is accomlished through a Convolutional Neural Network (CNN) architecture
in conjunction with PANet layers to improve feature representation (which is desirable for different features such as color, 
edges, or clustering that  could be traits of detectable objects). The primary approach used for pre-processing was batch
normalization to ensure that each layers inputs are stable (similar to what was done in the GOES Remote Detection ConvLSTM layers). 

Mentioned below were the recorded best recorded loss metrics: 
results_dict: {'metrics/precision(B)': 0.9151360241743095, 'metrics/recall(B)': 0.741217859008031, 
'metrics/mAP50(B)': 0.8355174905213206, 'metrics/mAP50-95(B)': 0.6504686874372195, 'fitness': 0.6689735677456297}

Precision: 0.9151360241743095
Recall: 0.741217859008031
mAP50: 0.8355174905213206
mAP50-95: 0.6504686874372195
Fitness: 0.6689735677456297

The largest constraint when training such a model is the computational cost. I was unable
to train the model on the entirety of the COCO dataset due to the number of computational units not
to mention the time cost of training. The accuracy has shown to improve as the model's inputs are increased
but as a hobbyist, the cost is a bit too high. In the future I would recommend segmented learning 
to specifically train each of the 3 detected objects. Such clustering can help make more accurate predictions
by focusing on the patters within each segment (in this case, each of the 3 objects).
