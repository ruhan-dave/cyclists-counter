# Bycicle Counter
UC Davis ECS 171 FInal Project

## Abstract

Group members:
- Timothy Blanton
- Sohan Patil
- Ru Han Wang
- Clément Weinreich

The goal of our project is to address the problem of adapting infrastructure to cyclist flows. On campus, for example, it is difficult to estimate the number of bicycle parking spaces needed for different areas and buildings. We therefore want to create an automatic detector and counter of cyclists using a Jetson Nano. Once our project is completed, it would be possible to capture data about the flow of cyclists at different locations, and to use this data to make predictions or estimates of cyclist flows. To complete this project, we plan to use transfer-learning on a pre-trained object detection model, which means re-train the classification layer of the deep convolutional neural network on a custom dataset in order to create our own cyclist detector. To do so, we plan to use the [Tensorflow 2 Object Detection API](https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/index.html), and use one of their pre-trained model which will be a deep convolutional neural network. The dataset we will use is the [Cyclist Dataset for Object Recognition](https://www.kaggle.com/datasets/semiemptyglass/cyclist-dataset) published by [1] in 2016. It contains 13.7k labeled images of size 2048 x 1024, recorded from a moving vehicle in the urban traffic of Beijing. The labels contain the positions of the bounding boxes around the cyclists in this format: `id center_x center_y width height`. Then, we plan to implement our own algorithmic-based tracking system that would allow us to count the number of cyclists detected over a period of time.

## Instructions for Cloning :

* To clone the repository including the dataset ( ~2GB stored with git LFS) :
```
git clone git@github.com:Clement-W/bycicle-counter.git
```
* To clone the repository without the LFS files (without the dataset) :
```
GIT_LFS_SKIP_SMUDGE=1 git clone git@github.com:Clement-W/bycicle-counter.git
```

## Data Exploration

## Data Preprocessing

## References

[1] X. Li, F. Flohr, Y. Yang, H. Xiong, M. Braun, S. Pan, K. Li and D. M. Gavrila. A New Benchmark for Vision-Based Cyclist Detection. In Proc. of the IEEE Intelligent Vehicles Symposium (IV), Gothenburg, Sweden, pp.1028-1033, 2016.

