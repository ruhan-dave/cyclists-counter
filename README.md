<!-- #region -->
# Bycicle Counter
UC Davis ECS 171 FInal Project

## Abstract

Group members:
- Timothy Blanton
- Sohan Patil
- Ru Han Wang
- Clément Weinreich

The goal of our project is to address the problem of adapting infrastructure to cyclist flows. On campus, for example, it is difficult to estimate the number of bicycle parking spaces needed for different areas and buildings. We therefore want to create an automatic detector and counter of cyclists using a Jetson Nano. Once our project is completed, it would be possible to capture data about the flow of cyclists at different locations, and to use this data to make predictions or estimates of cyclist flows. To complete this project, we plan to use transfer-learning on a pre-trained object detection model, which means re-train the classification layer of the deep convolutional neural network on a custom dataset in order to create our own cyclist detector. To do so, we plan to use the [Tensorflow 2 Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html), and use one of their pre-trained model which will be a deep convolutional neural network. The dataset we will use is the [Cyclist Dataset for Object Recognition](https://www.kaggle.com/datasets/semiemptyglass/cyclist-dataset) published by [1] in 2016. It contains 13.7k labeled images of size 2048 x 1024, recorded from a moving vehicle in the urban traffic of Beijing. The labels contain the positions of the bounding boxes around the cyclists in this format: `id center_x center_y width height`. Then, we plan to implement our own algorithmic-based tracking system that would allow us to count the number of cyclists detected over a period of time.

## Setup Requirements :

* Clone the repository :
```
git clone git@github.com:Clement-W/bicycle-counter.git
cd bicycle-counter
```

* Download the data :
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1u39ZCDroyUpguicPMUZ20eIeux2N7uql' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1u39ZCDroyUpguicPMUZ20eIeux2N7uql" -O images.zip && rm -rf /tmp/cookies.txt
```
The data can also be download manually [here](https://drive.google.com/file/d/1u39ZCDroyUpguicPMUZ20eIeux2N7uql/view?usp=sharing).

* Unzip and delete the zip file :
```
unzip images.zip
rm images.zip
```


Now you're all set!

## Data Exploration

The data exploration phase is split into 2 notebooks :
1. [adapt_dataset.ipynb](adapt_dataset.ipynb) 

The first notebook contains the steps to adapt the original dataset, and make it usable for our project. In fact, when we downloaded the dataset, we discovered that 1623 images out of 13674 were not labelled. They had an empty label `txt` file. Thus we decided to remove theses images, so the dataset was reduced to 12051 images. Thanks to the python library `pylabel`, we loaded the dataset as a dataframe in order to explore this new version of the dataset. This library was useful in particular to convert the label format from Yolov5 to VOC XML. So we removed the `.txt` label and keept the new `.xml` labels. After that, we used a script from the tensorflow object detection api to split our dataset into 3 sets : 
* 90% for the train set (9760 images)
* 10% for the test set (1206 images)
* 10% of the 90% train set for the validation set (1085 images)
<!-- #endregion -->

All these manipulations necessited to manage folders and files with command lines, that's why this notebook can't be run anymore. Before executing this notebook, the main folder looked like this:

* `images/` contains 13 674  images
* `labels/` contains 13 674 .txt files (yolo bounding box format) with this format `id center_x center_y width height`


Now, after running this notebook the main folder looks like this:

* `adapt_dataset.ipynb` this jupyter notebook
* `tensorflow-scripts/` contains the scripts from tensorflow
    * `partition_dataset.py` python script to split a folder of images with labels into 2 subfolders train and test
* `images/` contains the data
    * `train/` contains 9760 images and labels of the train set 
    * `test/` contains 1085 images and labels of the test set
    * `validation/` contains 1206 images and labels of the validation set


Now that the dataset is usable for the project, we can perform some data analysis on it.

2. [data_analysis.ipynb](data_analysis.ipynb)

Todo

## Data Preprocessing

## References

[1] X. Li, F. Flohr, Y. Yang, H. Xiong, M. Braun, S. Pan, K. Li and D. M. Gavrila. A New Benchmark for Vision-Based Cyclist Detection. In Proc. of the IEEE Intelligent Vehicles Symposium (IV), Gothenburg, Sweden, pp.1028-1033, 2016.

