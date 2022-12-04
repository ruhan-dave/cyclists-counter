<h1 style=text-align:center> Cyclists Counter - ECS171 Final Project </h1>


Clément Weinreich - Timothy Blanton - Sohan Patil - Dave Ru Han Wang


# Introduction

Introduction of your project. Why chosen? why is it cool? General/Broader impact of having a good predictive mode. i.e. why is this important?


# Methods

All the work that have been done for this project is reproductible. To download our dataset and install the required softwares and libraries, you can follow the instructions in [Setup-requirements](#Setup-requirements).


## Data Exploration

The data exploration step is split into 2 notebooks:
- [adapt_dataset.ipynb](adapt_dataset.ipynb): Adapt the original dataset and make it usable for our project.
- [data_exploration.ipynb](data_exploration.ipynb): Exploration of the dataset

###  [Adapt Dataset notebook](adapt_dataset.ipynb)

This first notebook contains the steps to adapt the original dataset ([Cyclist Dataset for Object Recognition](https://www.kaggle.com/datasets/semiemptyglass/cyclist-dataset)), and make it usable for our project. The original dataset contains 13674 images, and the labels consist of .txt files containing the class id followed by the coordinates : `id center_x center_y width height`. Thus, deleted the non-labeled images along with their empty label file. Thanks to the python library `pylabel`, we loaded the dataset into an object  This library was useful in particular to convert the label format from Yolov5 (.txt) to VOC XML (the required label format of tensorflow object detection API). After converting the lables, we removed the `.txt` labels and keept the new `.xml` labels. Then, we used a script from the tensorflow object detection API to split our dataset into 3 sets : 
* 90% for the train set (9760 images)
* 10% for the test set (1206 images)
* 10% of the 90% train set for the validation set (1085 images)

All these manipulations necessited to manage folders and files with command lines. To run this notebook, you must download the original dataset from kaggle before (see in the notebook). With the current state of the github repository, the notebook can't be run anymore. Before executing this notebook, the main folder looked like this:

* `images/` contains 13 674  images
* `labels/` contains 13 674 .txt files (yolo bounding box format) with this format `id center_x center_y width height`

After running this notebook the main folder looked like this:

* `adapt_dataset.ipynb` this jupyter notebook
* `tensorflow-scripts/` contains the scripts from tensorflow
    * `partition_dataset.py` python script to split a folder of images with labels into 2 subfolders train and test
* `images/` contains the data
    * `train/` contains 9760 images and labels of the train set 
    * `test/` contains 1206 images and labels of the test set
    * `validation/` contains 1085 images and labels of the validation set
    
To download the new version of the dataset, you can also directly follow the steps in [Setup-requirements](#Setup-requirements). 

### [Data Exploration notebook](data_exploration.ipynb)

This notebook contains all the work done to explore the dataset. In this notebook, we:
* Load the dataset as a pandas dataframe thanks to the pylabel library
* Analyze the the number of images, of bounding boxes
* Analyze the image size
* Analyze some statistics (mean,std,quantiles,min,max) about the repartitions of the bounding boxes
* Visualize some samples from train,test and validation set with the bounding boxes.
* Analyze the size and the repartition of small/medium/large bouding boxes in the dataset (based on the sizes used by the COCO standard metrics)


## Data Preprocessing

### Data preprocessing with tensorflow 2 object detection API

Now that the dataset has been prepared to be compatible with our project, and that we explored it a little bit more, we can focus at the data preprocessing we will setup in order to use the data during training.

The Tensorflow 2 Object Detection API allows to do data preprocessing and data augmentation in the training pipeline configuration. As stated in their [docs](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/configuring_jobs.md#configuring-the-trainer), all the preprocessing of the input is done in the `train_config` part of the training configuration file. The training configuration file is explained in the section [Configure the training pipeline](#TODODODODOOODODODODODOO) of this README.md. 

So here, the important part of the configuration file is `train_config` which parametrize:
* Model parameter initialization
* Input Preprocessing
* SGD parameters

Thus, we focus on the Input Preprocessing part of the config file. All this preprocessing is included in the `data_augmentation_options` tag of the `train_config`. This data_augmentation_options can take several values that are listed [here](https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto). And [this file](https://github.com/tensorflow/models/blob/master/research/object_detection/builders/preprocessor_builder_test.py) also explains how to write them into the config file. 

Concerning the image size, most of the object detection model contains an  [image-resize layer](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Resizing) which resizes the input to the required size. Thus, the images are automatically resized to the desired input size when feeded to the network. 

### Data Augmentation

The pipeline that specify the data augmentation options that will be applied to the images is created in this notebook: [configure_training_pipeline.ipynb](configure_training_pipeline.ipynb).

Data augmentation englobe techniques used to increase the amount of data, by adding to the dataset slightly modified copies of already existing data. Data augmentation helps to reduce overfitting by helping the network to generalize over different examples. This is closely related to oversampling. It is important to note that all the data augmentation options will be applied on the images, before entering the resize layer. Here, we used 3 different methods to augment our data:
* **random_scale_crop_and_pad_to_square**: Randomly scale, crop, and then pad the images to fixed square dimensions. The method sample a random_scale factor from a uniform distribution between scale_min and scale_max, and then resizes the image such that its maximum dimension is (output_size * random_scale). Then a square output_size crop is extracted from the resized image. Lastly, the cropped region is padded to the desired square output_size (which is the input size of the network) by filling the empty values with zeros.
* **random_horizontal_flip**: Randomly flips the image and detections horizontally, with a probability p. Here we chose p=0.3, so the probability that an image is horizontally flipped is 30%.
* **random_distort_color**: Randomly distorts color in images using a combination of brightness, hue, contrast and saturation changes. By using the parameter `color_ordering=1`, the sequence of adjustment performed is :
  1. randomly adjusting brightness
  2. randomly adjusting contrast
  3. randomly adjusting saturation 
  4. randomly adjusting hue.
  
In the training configuration file, this will look like this:
```py
  data_augmentation_options {
    random_horizontal_flip {
      probability: 0.3
    }
  }
  data_augmentation_options {
    random_scale_crop_and_pad_to_square {
      output_size: 640
      scale_min: 0.1
      scale_max: 2.0
    }
  }
  data_augmentation_options {
    random_distort_color {
      color_ordering: 1
    }
  }
```

  


<!-- #region -->
## Main model

### Configure the training job

To follow this part, we assume you installed the required libraries to train an object detection neural net with the tensorflow 2 object detection API. If not, just follow the instructions [here](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html). If you don't plan to train the model, you don't need to install these libraries.

#### The training workspace

To train our object detection model, we followed the [documentation](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/training.html) of the Tensorflow2 Object Detection API. Thus, we organized the training workspace `training-workspace` the same way as it is recommended:

* **annotations**: This folder is used to store all the TensorFlow *.record files, which contains the list of annotations for our dataset images.

* **exported-models**: This folder is used to store exported versions of our trained model(s).

* **models**: This folder contains a sub-folder for each of the training job. Each subfolder contains the training pipeline configuration file `pipeline.config`, as well as all files generated during the training and evaluation of our model (checkpoints, tfevents, etc.).

* **pre-trained-models**: This folder contains the downloaded pre-trained models, which shall be used as a starting checkpoint for our training jobs.

#### Generate .record files [(generate_tfrecords.ipynb)](generate_tfrecords.ipynb)

The Tensolfow API use what we call tf record files to store the data. It is a simple format that contains both the images and the labels in one file. To generate these files, we followed the documentation. The instructions are explained in the notebook generate_tfrecords.ipynb. In the end, this adds 3 new files to the folder `training-workspace/annotations`:
* `train.record`: the train set
* `validation.record`: the validation set
* `test.record`: the test set

The .record files are associated to a `label_map` file which tells the classes that must be classified in the dataset. Here we only want to classify the cyclists, so the label map is very simple:
```py
item {
    id: 1
    name: 'cyclist'
}
```
This label map is stored in `training-workspace/annotations/label_map.pbtxt` along with the .record files.

#### Download Pre-Trained Model [(download_pretrained_network.ipynb)](download_pretrained_network.ipynb)

The pre-trained object detection models of the tensorflow object detection API are listed [here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md). Many different architecture exists such as RCNN, Faster-RCNN, SSD, etc. We have chosen the EfficientNet architecture. The EfficientNet architecture has been proposed by [2] (EfficientNet: Rethinking Model Scaling for Convolutional Neural Network) in 2019. Here is the model architecture of the base EfficientNet model:

<img src="markdown-images/efficientDet-architecture.png" alt="model architecture" style="height: 200px"/>
<br/>

<div style='text-align:center'> <b>Table 1</b>: Architecture of the efficientNet base model </div>

This table (from the original paper [2]) shows that the main block of this architecture is the MBConv. The MBConv is an inverted linear bottleneck layer, with depth-wise separable convolution. A depth-wise separable convolution conist of splitting a normal $k\times k$ convolution in two "simpler" convolutions, to reduce the number of parameters, and speed up the compute time. The inverted linear bottleneck layer change the output of the last convolution of a classic residual block by a linear output, before it is added to the initial activation by the skip connection. This architecture has shown better results in accuracy and in computer preformance (in FLOPS).

We decided to use `EfficientDet D1 640x640` as our pre-trained object detector. The last checkpoint of the training and the training configuration file is available in the directory `training-workspace/pre-trained-models/efficientdet_d1_coco17_tpu-32`. To download this pre-trained model, you can follow the notebook [(download_pretrained_network.ipynb)](download_pretrained_network.ipynb).

#### Configure the training pipeline [(configure_training_pipeline.ipynb)](configure_training_pipeline.ipynb)


The TensorFlow Object Detection API uses protobuf files to configure the training and evaluation process. The config file is split into 5 parts: model, train_config, train_input_reader, eval_config and eval_input_reader. A complete description of each part can be found in the notebook [(configure_training_pipeline.ipynb)](configure_training_pipeline.ipynb).

To redefine the training configuration through the file `pipeline.config`, we created the directory `efficientdet_d1_v1` into `training-workspace/models`. Then, we copied the configuration file into this directory in order to modify it, while keeping the original configuration file in the folder `training-workspace/pre-trained-models/efficientdet_d1_coco17_tpu-32`. 

We described every changes done to the configuration file in the notebook, but here are the main changes:
* Changed the number of class to detect
* Changed the batch size
* Added some data augmentation options (see [Data-Augmentation](#Data-Augmentation)) 
* Changed the learning rate
* Changed the paths of the checkpoints and the dataset

Some parameters have not been changed, but remains important to mention, like the learning rate decay scheduler. In order to lower the learning rate as the training progresses, we used a cosine decay schedule. This schedule applies a cosine decay function to an optimizer step, given a provided initial learning rate. 

For more details about the training configuration file, check the notebook `configure_training_pipeline.ipynb`. You can also access the [pipeline.config](training-workspace/models/efficientdet_d1_v1/pipeline.config) manually.
<!-- #endregion -->

<!-- #region -->
### Train the model

The model was trained for 2 days (50 hours) on a NVIDIA 3060TI.

#### Launch training and periodic evaluation

To train the model, and include a periodic evaluation of the model with the validation set, 2 terminal are required:
* Train the model in the first terminal with
```sh
cd training-workspace
python ../tensorflow-scripts/model_main_tf2.py --model_dir=models/efficientdet_d1_v1 --pipeline_config_path=models/efficientdet_d1_v1/pipeline.config
```
* Launch periodic evaluation in the other terminal with
```sh
cd training-workspace
python ../tensorflow-scripts/model_main_tf2.py --model_dir=models/efficientdet_d1_v1 --pipeline_config_path=models/efficientdet_d1_v1/pipeline.config --checkpoint_dir=models/efficientdet_d1_v1
```

The model was trained for 2 days (50 hours) on a NVIDIA 3060TI. If you want to access the model directory which contains all the training checkpoints, the evaluation reports (as .tfevents files), you can download the folder `efficientdet_d1_v1`, and replace the current folder `training-workspace/models/efficientdet_d1_v1` by the new one. To download it:


* Download the efficientdet_d1_v1 folder :
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1ClnmwtBJbP6FIJNP2WWV8ZwfgWxiPFxX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1ClnmwtBJbP6FIJNP2WWV8ZwfgWxiPFxX" -O efficientdet_d1_v1.zip && rm -rf /tmp/cookies.txt
```
It can also be download manually [here](https://drive.google.com/file/d/1ClnmwtBJbP6FIJNP2WWV8ZwfgWxiPFxX/view?usp=sharing).

* Unzip and delete the zip file :
```
unzip efficientdet_d1_v1.zip
rm efficientdet_d1_v1.zip
```

You know have access to the result of the training phase.
<!-- #endregion -->

#### Monitor the training job using Tensorboard

To monitor how the training is going, a popular tool is [tensorboard](https://www.tensorflow.org/tensorboard). This tool is automatically installed with tensorflow. In case you just want tensorboard, you can still install it manually with `pip install tensorboard`. We will go over the details of the training in the next section [TODODODODODAnalysis of the training phase](#analysis-of-the-training-phase). We uploaded our instance of tensorboard, so you can access it on [that link](https://tensorboard.dev/experiment/LMZdMvMxTwGcZ0ypzM7WGg/). Unfortunately, we can only publish the time series graphs, so you will not be able to see the image examples. But having that link open while reading the [Results section](#Results), can help you to better understand the graphs, in particular by changing the smoothing cursor.

If you have downloaded the whole efficientdet_d1_v1 directory, (see [Launch training and periodic evaluation](#Launch-training-and-periodic-evaluation)), then you can open tensorboard in our model directory with this command:
```
tensorboard --logdir=efficientdet_d1_v1
```
You should then be able to review all the information of the training, including the image examples from data augmentation, and the predictions in the validation set.


#### Focus on the loss function used

As we are training an object detector, there is 4 loss that we need to take into account:

* The classification loss: Is the cyclists well classified as cyclists
* The localization loss: Is the bounding boxes close to the cyclists
* The regularization loss: Aims to keep the model parameters as small as possible
* The total loss: Sum of the classification, localization, and regularization loss.

#### Focus on the performance metrics used

During the training, an evaluation of the last checkpoint is performed every hour. During this evaluation, the evaluation loss was computed, but also the performance with the mAP (mean Average Precision) and mAR (mean Average Recall) metrics. These two metrics are popular to measure the accuracy of object detection models. These metrics uses the concept of IoU (Intersection over Union). The IoU measure the overlap between 2 boundaries. The IoU measure how much the predicted bounding box overlaps with the actual bounding box that must be predicted. Usually, we define the IoU threshold to 0.5, so if the IoU is equal or greater than 0.5, then we say that the prediction is a true positive, else it is a false positive. Thus, in the `pipeline.config` file, the IoU threshold is defined at 0.5.

To compute the mAP, we first need to compute the precision and recall of detecting the bounding boxes correctly (thanks to the IoU) for each images. Then, we can plot a graph of precision vs recall. To compute the AP, we just need to compute the area under the precision-recall curve, using an interpolation technique. The COCO mAP impose to do a 101-point interpolation to calculate the precision at the 101 equally spaced recall level, and then average them out. To have the mAP, we then need to take the mean over all classes, but as we only have one class here, the mAP is equivalent to the AP.

Then we also use the AR (Average Recall), which consist of averaging the recall at IoU thresholds from 0.5 to 1, and thus summarize the distribution of recall across a range of IoU thresholds. So we can plot the recall values for each IoU threshold between 0.5 and 1. Then, the average recall describes the area doubled under the recall-IoU curve. Similarly to mAP, the mAR take the mean of the AR for every class. So the mAR is here equivalent to the AR.

We can monitor these metrics in tensorboard and have an insight of the model's performance according to different cases. For the average precision we measured:
* The mAP at IoU varying from 0.5 to 0.95 (coco challenge metric) -> named mAP in tensorboard
* The mAP at IoU = 0.5 (PASCAL VOC challenge metric) -> named mAP@.50IOU in tensorboard
* The mAP at IoU = 0.75 (strict metric) -> named mAP@.75IOU in tensorboard
* The mAP for small objects (area < $32^2$) -> named mAP(small) in tensorboard
* The mAP for medium objects ( $32^2$ < area < $96^2$) -> named mAP(medium) in tensorboard
* The mAP for large objects (area > $96^2$) -> named mAP(large) in tensorboard

For the average recall we measured:
* The AR given images with 1 detection maximum -> named AR@1 in tensorboard
* The AR given images with 10 detection maximum -> named AR@10 in tensorboard
* The AR given images with 100 detection maximum -> named AR@100 in tensorboard
* The AR for small objects (area < $32^2$) -> named AR@100(small) in tensorboard
* The AR for medium objects ( $32^2$ < area < $96^2$) -> named AR@100(medium) in tensorboard
* The AR for large objects (area > $96^2$) -> named AR@100(large) in tensorboard

<!-- #region -->
### Evaluate the final model

To test the model, we used a similar approach as what we did for training. First, we created a new folder `models/efficientdet_d1_v1_test` which contains the last checkpoint of the training, and a copy of the pipeline.config file. We evaluated the model on the whole test, validation and training set. This gave us 3 scalars for each performance metrics, and loss values. Then, we decided to export those results to a pandas dataframe. The only way we found to do this, is to export the folder `models/efficientdet_d1_v1_test` to the tensorboard dev API, and then import the data as a pandas dataframe. To do so, we first needed to create a new project in the tensorboard dev API:
```sh
tensorboard dev upload --logdir training-workspace/models/efficientdet_d1_v1_test
```

In a new terminal, we then ran the evaluations on each set. To run the evaluation with the test set, we did:

- In the eval_input_reader part of the copy of pipeline.config, we changed input_path to `annotations/test.record`.
- Then we launched the evaluation command (which is the same as the one we used during training):
```sh
cd training-workspace
python ../tensorflow-scripts/model_main_tf2.py --model_dir=models/efficientdet_d1_v1_test --pipeline_config_path=models/efficientdet_d1_v1_test/pipeline.config --checkpoint_dir=models/efficientdet_d1_v1_test
```
- A folder named eval has been created in `models/efficientdet_d1_v1_test`, which contains the .tfevents file corresponding to the evaluation of the test set. Thus, we moved it in its corresponding folder:
```sh
cd models/efficientdet_d1_v1_test
mkdir test
mv eval/*.tfevents test
```
The, we repeted these steps for the training and validation set, just by replacing the input_path in the pipeline by `annotations/validation.record` or `annotations/train.record`, and in the last step, create the folders `validation` and `training` and move the .tfevents files accordingly to their respective folder.

Once it is done, we have the 3 evaluation folders `models/efficientdet_d1_v1_test/training`,`models/efficientdet_d1_v1_test/test` and `models/efficientdet_d1_v1_test/validation`. We also have an empty `models/efficientdet_d1_v1_test/eval` folder that we deleted: `rmdir models/efficientdet_d1_v1_test/eval`.

To do the conversion of the results from `*.tfevents` into a pandas dataframe, we used the tensorboard library in python. This work can be found in the [evaluate_model.ipynb](evaluate_model.ipynb) notebook.

If you want to download the folder `efficientdet_d1_v1_test` which contains the evaluations results as .tfevents files, you can do it by following these steps:

* Download the `efficientdet_d1_v1_test` folder :
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Q_usxj2k0waHw4-6epKXPxcdn_GG3D1f' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Q_usxj2k0waHw4-6epKXPxcdn_GG3D1f" -O efficientdet_d1_v1_test.zip && rm -rf /tmp/cookies.txt
```
It can also be download manually [here](https://drive.google.com/file/d/1Q_usxj2k0waHw4-6epKXPxcdn_GG3D1f/view?usp=sharing).

* Unzip, delete the zip file, and move the folder at its proper emplacement :
```
unzip efficientdet_d1_v1_test.zip
rm efficientdet_d1_v1_test.zip
mv efficientdet_d1_v1_test training-workspace/models/
```
<!-- #endregion -->

### Export the model

Now that the model is trained, we export it as a .pb file. To do so, we can use the tensorflow script `exporter_main_v2.py` and execute:
```
python tensorflow-scripts/exporter_main_v2.py --input_type image_tensor --pipeline_config_path training-workspace/models/efficientdet_d1_v1/pipeline.config --trained_checkpoint_dir training-workspace/models/efficientdet_d1_v1 --output_directory training-workspace/exported-models/my_model
```
Now, we can use this model to perform inference.

If you want to download the exported model, you can do it that way:

* Download the my_model folder :
```
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1BWaKL_VMOQ89Nuw5NEdRUv4Jl6GP6trC' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1BWaKL_VMOQ89Nuw5NEdRUv4Jl6GP6trC" -O my_model.zip && rm -rf /tmp/cookies.txt
```
It can also be download manually [here](https://drive.google.com/file/d/1BWaKL_VMOQ89Nuw5NEdRUv4Jl6GP6trC/view?usp=sharing).

* Unzip, delete the zip file, and move the exported model at its proper place :
```
unzip my_model.zip
rm my_model.zip
mv my_model training-workspace/exported-models/
```


### Inference with the model

#### Inference on the Jetson Nano

TODO: describe the method (not the results)

#### Inference in Hugging Face Spaces

We created a hugging face space in order to allow everyone play, and see the results of our object detection model. You can access it on that link: [cyclists-detection](https://huggingface.co/spaces/clement-w/cyclists-detection). You can upload images by URL, or from your computer.


## Smaller model

TODO: Say that it is the exact same procedure, but we download a new model, and say that we did not added any data augmentation or anything


# Results


## Data Exploration

In this section, we only describe the main results obtained from the data exploration phase. For more details, you can read the content of the [Data Exploration notebook](data_exploration.ipynb). The dataset contains 12051 images with 9760 for training, 1206 for testing and 1085 for validation. The only class of objects is `cyclist`. Each images have the same size: 2048x1024x3 (3 because of the RGB color channel). There is a total of 22173 bounding box in the dataset, so 22173 cyclists labeled. Most of the bounding boxes have a low width (between 25px and 200px). In the dataset, the average centroid of the bounding boxes is (1104x470) which almost corresponds to the center of the image. There is 40% of small bounding boxes, 40% of medium bounding boxes, and 20% of large bounding boxes. There is approximatively 2.3 bounding boxes per image, and the median number of bounding boxes per image is 2. Finally, when visualizing some images from the dataset, we noticed that they all have a low luminosity and a low contrast.

<!-- #region -->
## Data Preprocessing

During the training phase, we can monitor many parameters with Tensorboard. Tensorboard also offers the possibility to see some examples of input images after data augmentation. Here are 3 of them:

|                                                      Example 1                                                      	|                                                      Example 2                                                      	|                                                      Example 3                                                      	|
|:-------------------------------------------------------------------------------------------------------------------:	|:-------------------------------------------------------------------------------------------------------------------:	|:-------------------------------------------------------------------------------------------------------------------:	|
| <img src="markdown-images/data-augmentation1.png" alt="img example after data augmentation" style="height: 200px"/> 	| <img src="markdown-images/data-augmentation2.png" alt="img example after data augmentation" style="height: 200px"/> 	| <img src="markdown-images/data-augmentation3.png" alt="img example after data augmentation" style="height: 200px"/> 	|

<br/>
<div style='text-align:center'> <b>Figure 1</b>: Example of data augmented images </div>


These 3 examples shows images where the color have been distorded. And the first example have been randomly cropped and padded to a square (we can guess that it was the left part of the image).
<!-- #endregion -->

## Training of the main model

<!-- #region -->
### Learning rate decay

The model was trained for 2 days. The training stopped automatically thanks to the cosine decay. Thanks to tensorboard, we can see how the learning rate has changed over the training with respect to the number of steps, or the time of training:

|                                  By number of steps                                  	|                                       By time                                      	|
|:------------------------------------------------------------------------------------:	|:----------------------------------------------------------------------------------:	|
| <img src="markdown-images/lr-steps.png" alt="lr w.r.t steps" style="height: 200px"/> 	| <img src="markdown-images/lr-time.png" alt="lr w.r.t time" style="height: 200px"/> 	|

<br/>
<div style='text-align:center'> <b>Figure 2</b>: Plot of the learning rate versus the number of steps, and the time </div>


We notice that at some point, the learning rate was equal to 0, which has stopped the training.

### Monitoring the loss

Here are the 3 most relevant loss that we can analyze. The loss on the training set in orange, and the loss on the validation set is blue:

|                                              Classification loss                                              	|                                             Localization loss                                             	|                                          Total loss                                         	|
|:-------------------------------------------------------------------------------------------------------------:	|:---------------------------------------------------------------------------------------------------------:	|:-------------------------------------------------------------------------------------------:	|
| <img src="markdown-images/classification-loss.png" alt="graph of classification loss" style="height: 200px"/> 	| <img src="markdown-images/localization-loss.png" alt="graph of localization loss" style="height: 200px"/> 	| <img src="markdown-images/total-loss.png" alt="graph of total loss" style="height: 200px"/> 	|

<br/>
<div style='text-align:center'> <b>Figure 3</b>: Plot of the 3 main loss curves versus the number of step </div>


The above curves are smoothed to make it easier to distinguish the tendency, but the actual curve is also displayed on the same graph with a low opacity.

### Monitoring the performances

Here are the results we obtained on the mean Average Precision:

<img src="markdown-images/mAP.png" alt="6 curves of the mAP" style="height: 400px"/>
<br/>
<div style='text-align:center'> <b>Figure 4</b>: Plot of the mAP performance metrics versus the number of step </div>

Here are the results we obtained on the Average Recall:

<img src="markdown-images/AR.png" alt="6 curves of the AR" style="height: 400px"/>
<br/>
<div style='text-align:center'> <b>Figure 5</b>: Plot of the AR performance metrics versus the number of step </div>

### Visualize some predictions during model evaluation

When the model is evaluated, tensorboard allow us to see 10 images with our model predictions. Here are 4 examples coming from different stage of training:

|     Stage of training     	|                                  Prediction VS Ground truth                                  	|
|:-------------------------:	|:--------------------------------------------------------------------------------------------:	|
|      First evaluation     	| <img src="markdown-images/eval1.png" alt="img example of prediction" style="height: 200px"/> 	|
| Second to last evaluation 	| <img src="markdown-images/eval4.png" alt="img example of prediction" style="height: 200px"/> 	|
| Last evaluation           	| <img src="markdown-images/eval2.png" alt="img example of prediction" style="height: 200px"/> 	|
| Last evaluation           	| <img src="markdown-images/eval3.png" alt="img example of prediction" style="height: 200px"/> 	|

<br/>
<div style='text-align:center'> <b>Figure 6</b>: Predicted samples and ground truth at different evaluation step </div>
<!-- #endregion -->

## Final evaluation of the main model

The results of the evaluation of the 3 set of data can be seen online on the [associated tensorboard](https://tensorboard.dev/experiment/dTD0vaI3SdyRZYI4WLHRbg/#scalars&runSelectionState=eyJldmFsIjpmYWxzZX0%3D) but this is not so relevant as it is just data point. The full results of model evaluation can be seen in the [evaluate_model.ipynb](evaluate_model.ipynb) notebook. 

Here are the obtained losses on the 3 sets:

|           	| Classification loss 	| Localization loss 	| Total loss 	|
|-----------	|:-------------------:	|:-----------------:	|------------	|
| Test set  	| 0.223413            	| 0.003314          	| 0.264104   	|
| Train set 	| 0.181020            	| 0.003023          	| 0.221419   	|
| Test set  	| 0.197278            	| 0.003231          	| 0.237886   	|

<br/>
<div style='text-align:center'> <b>Table 2</b>: Loss obtained on the 3 sets </div>

Here are 3 of the performance metrics obtained on the 3 sets:

|           	| mAP@.50IOU 	| map@.75IOU 	|  AR@100  	|
|-----------	|:----------:	|------------	|:--------:	|
| Test set  	| 0.823290   	| 0.649656   	| 0.658216 	|
| Train set 	| 0.840275   	| 0.676805   	| 0.672312 	|
| Test set  	| 0.826129   	| 0.655632   	| 0.662531 	|

<br/>
<div style='text-align:center'> <b>Table 3</b>: mAP and AR obtained on the 3 sets </div>


## Training and evaluation of the second model

TODO (not as detailled as the main model)

<!-- #region -->
## Inference with the main model

### Inference in the hugging face space

Here are some of the results we had when running our model in the hugging face space:

TODO


### Results on a video flux

TODO

### Results on the Jetson Nano

TODO


<!-- #endregion -->

# Discussion


## Data Exploration

In this section, we describe and interpret the main results we had during data exploration.

First, most of the bounding boxes have a low width (between 25 and 200px). The height of the bounding boxes is a bit more spread out. Furthermore, 75% of the bounding boxes have a height less than 285px. This is very informative because it shows that a large percentage of the bounding boxes are small-moderate. It means that a lot of images will contains cyclists that are far away from the camera, or at least not close by.

We also noticed that the bounding boxes are mostly in the center, but still a large part of the bouding boxes are spread out on all the width of the image. In the contrary, the y coordinate of the centroid of the bounding boxes are more compressed between 400 and 600 px, so in the middle of the y axis. This is expected because if the camera is on the road level, every cyclists will be in front of the camera, so in the middle of the y axis.

By analyzing the proportion of small/medium/large bounding boxes, we confirm that 80% of the bounding boxes are small/medium, and only 20% are large bounding boxes. As the small/moderate examples are the hardest to detect, it is good to know that the model is trained on more examples with complex situations. 

Finally, 50% of the images contains more than 2 bounding boxes. This is a relevant information for our project as we want the model to be able to recognize multiple cyclists at the same time in order to count them. 


## Data Preprocessing

In this project, the data preprocessing was mixed with the data exploration, because the original dataset needed to be modified in order to be compatible with the task we want to perform. We used the pylabel library to do every changes to the dataset on the file/folder level at first (label format change, split into 3 sets). This choice has been made to facilitate the reproducibility of our project. Managing folders with thousands of images is a delicate process, which is why we decided to do it all at first, and then explore the data. This choice can be criticized as it would have been better to explore the dataset before splitting it into 3 different set. 

Regarding the dataset we used, it is also very questionable to split the dataset randomly into train/test/validation sets. The dataset consists of images recorded from a moving vehicle in the urban traffic of Beijing. It means that some images are sequentials. Even they are all different, they can come from the same scene. Thus, when splitting the dataset randomly, we have a memory leakage between the 3 different sets. We noticed this fact in an advanced stage of the project, which is why we did not change our dataset.

As the images have a width of 2048 and a height of 1024, it was not feasable for us to train a neural network with these dimensions. But thanks to the resize layer of the object detection models we used, we are able to feed any input size to the network. 

Finally, all the data preprocessing that have been performed on the adapted dataset only consists of data augmentation:

* We choosed to use the random scale,crop and pad to square data augmentation option because during the data exploration phase, we noticed that most of the cyclists were in the center of the image. Thus, to give different examples to the model, this data augmentation option will create other images where the cyclists won't be in the center of the image.

* We choosed to use the horizontal_flip data augmentation option because this will create more examples to train the network. As the cyclists can come from the left,right or front of the camera, this data augmentation option will help the network to see diverse cases of cyclist positions.

* We choosed to use the random_distort_color data augmentation option because during the data exploration phase, we noticed that the luminosity of the images are low, with a low contrast and a low saturation. Thus, this data augmentation option will help the network to see other examples with a different brightness, contrast, saturation and hue.

<!-- #region -->
## Preparation of the main model

### Choice of pre-trained model

Today, it is EfficientNet based models (EfficientDet) that provide the best overall performances, and can work well for low latency applications. For example, `EfficientDet D1 640x640` can perform inference in 54ms on a nvidia-tesla-v100GPU, and obtain a COCO mAP of 38.4. As we aim to work on a live application on the jetson nano, this pre-trained model offers the best compromise between inference time and performances.


### Configuration of the training

First, we changed different hyperparameters. We had to decrease the batch size to 3 in order to fit the memory of the GPU used to train the network.E ven though it is a mini-batch gradient descent, the batch size remains very small compared to the dataset size, which is why it is very similar to a vanilla stochastic gradient descent. We also changed the learning rate from 0.04 to 0.02. At first, we used a learning rate of 0.04 but this has caused the gradient to explode. Thus, we decreased it until having a value which is not causing an exploding gradient.
<!-- #endregion -->

## Training of the main model

### Comments on the loss

As the batch size is very small, the loss is more subject to noise and vary a lot depending on the data contained in the batches. Thus, our loss curves consists of spikes which hardly show the tendency of these curves. To counter this, tensorboard offer the possibility to smooth the curves in order to make some curves more intelligible. These curves are displayed in **Figure 3**.

As you can see in low opacity, it is not so easy to discern the tendency of the loss curves. First, we can notice that in the 3 graphs, our training loss and validation loss are very close, so we can suppose that our model is not overfitting the training data. More particularly:

* The classification loss is decreasing in both training and validation set, so we can suppose that letting the training continue would have helped the network to better learn to classify the cyclists.
* The localization loss is very close to 0, and is close to be on a plateau. We can suppose that letting the training continue would have led to network to overfit the localization of the bounding boxes.
* The total loss was of course deacreasing as it is the sum of the two loss above.

Based on the error at each step, we can say that the model is fitting well the data from 50k steps, to the final step (300k). The fitting is still better on the last step as the loss is smaller. We can interpret these plots as seeing the complexity of the model and the error. Here, we can say that even though the model become more complex across the time, it is still fitting well the data. If we had more time to train the network, we could restart it on the last checkpoint, with a new base learning rate and see if the loss is still decreasing. This could maybe lead to better result, or cause overfitting. But this would probably mean to let the GPU run for 2 more days, which is costly in energy consumption. 

### Comments on the performance metrics

Let's start by discussing the mean average precision metrics, based on the results displayed in **Figure 4**.

Overall, the mAP is increasing no matter which value of IoU is used, or the size of the objects. The first plot show the COCO mAP (averaging on different values of IoU between 0.5 and 0.95), which is increasing but was starting to get on a plateau after 250k steps. We notice that the value of the mAP\@0.50IOU is larger than the value of the mAP\@0.75IOU, which is expected as 0.75 is a much more strict threshold. The mAP\@.75IOU on the last evaluation is approximately equal to 0.65, which is still very good.

To analyze the mAP according to different size of bounding boxes, we can refer to the data exploration notebook which contains the distribution of the width, and height of the bounding boxes. To know the proportion of small, medium and large bounding boxes in our dataset, we modified the [data exploration notebook](data_exploration.ipynb). Thus, we know that our dataset contains:
* 40% of small bounding boxes
* 40% of medium bounding boxes
* 20% of large bounding boxes

To better understand what small/medium/large means, we also displayed examples containing different size of bounding box. This really helps to understand and interpret the results.

Knowing this, we can know better analyze the mAP according to the size of the bounding box. First we notice that the mAP for the small bounding box is close to 0. Knowing that 40% of bounding boxes are small, we can't say that it is because of the lack of data. If we look at what small bounding boxes looks like in the data exploration notebook, we can understand how it is difficult to detect these bounding boxes and recognize a cyclist. Thus, to use the model in order to count the cyclists, the camera needs to be closer than the examples showing small bounding boxes. Now if we look at the medium bounding boxes, the mAP on the last evaluation has reached 0.4 which is satisfying. If we look at what medium bounding boxes looks like in the data exploration notebook, we notice that some bounding boxes remains very close to the small ones. Then, if we look at the large bounding boxes, the mAP on the last evaluation has reached 0.8 which is very accurate. Thus, we can look at some examples containing large bounding boxes, and set up the camera at similar distances to count the cyclists more effectively.

These results concerns the mean average precision, so contains mostly information about the proportion of true positive, so the percentage of the predictions that are correct. Now let's look at the Average Recall in **Figure 5**, which contains information about how good the bounding boxes are retrieved. 

Again, the AR is increasing in every cases. The AR for images containing 1 bounding box reached 0.4 on the last evaluation. The AR for images containing at most 10 bounding boxes, the AR reached 0.64 which means that overall the true positive are well retrieved. The results of AR@10 and AR for images containing at most 100 bounding boxes are approximatively equivalent. This can be explained because there is not many images that have more than 10 bounding boxes. In the data_exploration notebook, we also added the frequency of number of bounding boxes present in the images. Thus, we know that only 42 images have more than 10 bounding boxes in the whole dataset. As it is evaluated on the validation set (1085 images), we can suppose that the number of images having more than 10 bounding boxes is way lesser than 42. So AR@10 and AR@100 are approximatively equivalent. 

If we focus on the AR depending on the size of the image, the results are very good. The AR for large images reached 0.84 on the last evaluation. The AR for medium images reached 0.55 on the last evaluation. The AR for small images reached 0.16 on the last evaluation. These also show that closer the cyclysts are to the camera, better the results. But it also show that in general, the bounding boxes are well retrieved by the model, which is positive.


## Evaluation of the main model

Let's start by analyzing the loss obtained when evaluating the model on the 3 sets (see **Table 2**). The validation and train loss are pretty close. The test loss is a bit higher (0.04 higher than the train loss). Even though this difference is very small, we could think that it is due to overfitting. According to the loss curves presented in [Monitoring-the-loss](#Monitoring-the-loss), the evaluation on validation set was very close to the training set during the whole training, which is not an indication of overfitting. So the fitting graph was not showing a sign of overfitting. In addition, the validation set have a loss close to the training set. As the validation set has never been used to update the weights during training (the gradients are not computed during evaluation), and not used to tune the hyperparameters, this shows that the model generalized well enough, without overfitting the training data. If we focus more on the resuts of the evaluation on test set, we notice that this difference of loss is mainly due to the classification loss which is 0.04 higher than the train set. Thus, this difference can be due to the data in the test set which can be slightly harder to classify, with harder examples.

Now let's focus on 3 performance metrics showed in **Table 3**. The results are very good, we have an mAP with an IoU threshold of 0.5 (standard for pascal vox challenge) greater than 80\% for every set. If we use a more strict threshold (0.75), we still have an mAP greater than 60\% for every set, with the test set at ~0.65 which is very good. If we look at the average recall, we also have results greater than 65\% for every set. Thus, we can affirm that our model generalized well the dataset, and is good at detecting cyclists.


## Comparison with the second model

TODO


# Conclusion

TODO

<!-- #region -->
# Collaboration

#### Clément Weinreich : Project Leader

Contributions:
* Project lead 
* Adaptation of the dataset (code + write up)
* Data exploration (code + write up)
* Data preprocessing (code + write up)
* Configuration of training job (code + write up)
* Analysis of the training phase and results (write up)
* Comments on model evaluation (write up)
* Create demo with hugging face space (code + write up)
* TODO ADD THINGS

#### Timothy Blanton : Model trainer

Contributions:
* Trained the main model (code)
* Trained the second model (code)
* Evaluated the models (code)
* Export of the main model (code)
* TODO ADD THINGS


#### Sohan Patil : Jetson Nanoer?

Contributions:
* Setup of the jetson nano
* TODO ADD THINGS

#### Dave Ru Han Wang : Coder? Counter?

Contributions:
* Data exploration (code)
* TODO ADD THINGS
<!-- #endregion -->

# Extra-information


## Setup requirements

Follow the steps below, or clone the repository and run the notebook [setup_install.ipynb](setup_install.ipynb).

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

* [Optional if you don't want to train a model] Install every dependences necessary to train an object detection neural net with tensorflow object detection API. To do so, you can follow the installation instructions from the [official tensorlow object detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html).

Now you're all set!



## Counting algorithm

TODO: describe the method (not the results)
