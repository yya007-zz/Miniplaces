# scene-recognition-cnn
Collaborator: Zhengjia Huang

## problem description
The goal of this challenge is to identify the scene category depicted in a photograph. The data for this task comes from the Places2 dataset which contains 10+ million images belonging to 400+ unique scene categories. Specifically, the mini challenge data for 6.869 will be a subsample of the above data, consisting of 100,000 images for training, 10,000 images for validation and 10,000 images for testing coming from 100 scene categories. The images will be resized to 128x128 to make the data more manageable. Further, while the end goal is scene recognition, a subset of the data will contain object labels that might be helpful to build better models.

For each image, algorithms will produce a list of at most 5 scene categories in descending order of confidence. The quality of a labeling will be evaluated based on the label that best matches the ground truth label for the image. The idea is to allow an algorithm to identify multiple scene categories in an image given that many environments have multi-labels (e.g. a bar can also be a restaurant) and that humans often describe a place using different words (e.g. forest path, forest, woods). The exact details of the evaluation are available on [here](http://places2.csail.mit.edu/challenge.html). Original information can be found [here](http://6.869.csail.mit.edu/fa16/project.html).

## data set
Data set can be downloaded [here](http://6.869.csail.mit.edu/fa16/challenge/data.tar.gz) 

## model
ban_res_2.py achieves the best performance up to now, which is 70% top5 accuracy (see ban-15000)

## dependencies
[Tensorflow](https://www.tensorflow.org/)

[Nvidia cuda](https://developer.nvidia.com/cuda-toolkit)

## server support
[tEp](https://tep.mit.edu)
