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

* [Optional if you don't want to train a model] Install every dependences necessary to train an object detection neural net with tensorflow object detection API. To do so, you can follow the installation instructions from the [official tensorlow object detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html)

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


After running this notebook the main folder looks like this:

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

For our project, the data preprocessing was mixed with the data exploration, because the dataset needed to be modified in order to be compatible with the task we want to perform. But now that the dataset has been prepared, and that we explored it a little bit more, we can take a look at the data preprocessing we will setup in order to use the data during training.

The Tensorflow 2 Object Detection API allows to do data preprocessing and data augmentation in the training pipeline configuration. As stated in their [docs](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md#configuring-the-trainer), all this is done in the `train_config` part of the training configuration file. The training configuration file is explained in the section [Configure the training pipeline](#configure-the-training-pipeline) of this file. 

So here, the important part of the configuration file is `train_config` which parametrize:
* Model parameter initialization
* Input Preprocessing
* SGD parameters

Here we will focus on the input preprocessing part. All this preprocessing is included in the `data_augmentation_options` tag of the `train_config`. This data_augmentation_options can take several values that are listed [here](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto). And [This file](https://github.com/tensorflow/models/blob/master/research/object_detection/builders/preprocessor_builder_test.py) also explains how to write them into the config file. The most important preprocessing options that we'll use is `resize_image`.Our images are very big (2048x1024), so we'll downscale them to be compatible with the pre-trained network we'll use. Usually, the input size of these networks are 256x256. Thus, the `data_agumentation_options` tag will have to include 
``` 
resize_image {
      new_height: 256
      new_width: 256
}
```
Then, we will also add options to perform some data augmentations.

## Configure the training job

To follow this part, we assume you installed the required libraries to train an object detection neural net with the tensorflow 2 object detection API. If not, just follow the instructions [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).

### The training workspace

To train our object detection model, we followed the [documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html) of the Tensorflow2 Object Detection API. Thus, we organized the training workspace `training-workspace` the same way as it is recommended:

* annotations: This folder will be used to store all *.csv files and the respective TensorFlow *.record files, which contain the list of annotations for our dataset images.

* exported-models: This folder will be used to store exported versions of our trained model(s).

* models: This folder will contain a sub-folder for each of training job. Each subfolder will contain the training pipeline configuration file *.config, as well as all files generated during the training and evaluation of our model.

* pre-trained-models: This folder will contain the downloaded pre-trained models, which shall be used as a starting checkpoint for our training jobs.

### Generate .record files

The Tensolfow API use what we call tf record files to store the data. It is a simple format that contains both the images and the labels. To generate these files, we followed the documentation. Everything is explained in the notebook generate_tfrecords.ipynb

### Download Pre-Trained Model

[follow docs](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#download-pre-trained-model)

### Configure the training pipeline

[follow docs](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#configure-the-training-pipeline)

The TensorFlow Object Detection API uses protobuf files to configure the training and evaluation process. The config file is split into 5 parts:
* The `model` configuration. This defines what type of model will be trained (ie. meta-architecture, feature extractor). This will not be modified as we'll use a pre-trained network, we won't modify it's architecture.
* The `train_config`, which decides what parameters should be used to train model parameters (ie. SGD parameters, input preprocessing and feature extractor initialization values). This part is very important as we can preprocess our data and do some data augmentation.
* The `eval_config`, which determines what set of metrics will be reported for evaluation. 
* The `train_input_config`, which defines what dataset the model should be trained on.
* The `eval_input_config`, which defines what dataset the model will be evaluated on. That's why we created a validation set for our data.



## Train the model

[follow docs](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#training-the-model)

## Monitor the training job using Tensorboard

[follow docs](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#monitor-training-job-progress-using-tensorboard)

## Evaluate the model on Test data

[follow docs](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#evaluating-the-model-optional)

## Export the model

[follow docs](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html#monitor-training-job-progress-using-tensorboard)

## Counting algorithm

TODO

## Setup of the Jetson Nano

TODO

## Inference on Jetson Nano

TODO

## Results

TODO

## References

[1] X. Li, F. Flohr, Y. Yang, H. Xiong, M. Braun, S. Pan, K. Li and D. M. Gavrila. A New Benchmark for Vision-Based Cyclist Detection. In Proc. of the IEEE Intelligent Vehicles Symposium (IV), Gothenburg, Sweden, pp.1028-1033, 2016.

