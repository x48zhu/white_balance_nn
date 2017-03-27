# White Balance (Illuminant Estimation) Using Neural Network

## Motivation & Goal
The color of illumination affects the color of object surfaces in a picture. The goal is to estimate the illuminant color based on characteristic of the picture, from where we can recover the surface colors under white illumination.

The raw images are in RGB-color-space, and we convert/ normalize them into rg-color-space by
> r = R / (R + G + B)
> g = G / (R + G + B)

such that the blue chromaticity is 
> b = 1 - r - g

To convert PNG images in RGB-color-space into rg-color-space, see *image_norm_matlab/generate.m*.

The output of any neural network under this task should be two numbers, representing the estimation on illumination color in rg-color-space

## Models
There are three models experimented in this repository.

### Fully connected model
This model builds up a simple fully connected nueral network. It takes the features (dimension of 8) from Cheng-Prasad-Brown.

The network is composed of two hidden layers between input layer and output layer. This network is simple and easy to train. The size of hidden layers is easy to tune by arguments as follows:

`python -m simple.train --hidden1=6 --hidden2=4 --max_steps=10 --debug`

More arguments should be seen at the bottom of *simple/train.py*.

### Convolutional neural network
This model follows [Single and Multiple Illuminant Estimation Using Convolutional Neural Networks](https://arxiv.org/pdf/1508.00998.pdf).

The code doesn't fully implement the paper, as the local to global aggregation is not implemented. The current code simply take the median of local estimations.

To run the training job for this model, use:

`python -m cnn.train --model=single`

More arguments should be seen at the bottom of *cnn/train.py*.

### Convolutional neural network (two branch)
This model follows [Deep Specialized Network for Illuminant Estimation](http://personal.ie.cuhk.edu.hk/~ccloy/files/eccv_2016_illuminant.pdf). It builds up an ensemble leanring by spliting the fully connection layer into two independent branches that produces two estimation, and have a second network trained to select from results of these branches.

To run the training job for this model, use:

`python -m cnn.train --model=multiple`

More arguments should be seen at the bottom of *cnn/train.py*.

## Others
* Change the directory in `constants.py` accordingly.
* There is unit test (and any further tests should be addressed) in *test.py*. The python library *unittest* is used. To run the unit test:
`python -m unittest test.TestMethods`

## TODO
* Consider use Keras library to simplify the code. Keras makes defining and tuning the model easier. Also see Keras Callback for early stop.
* Check if the patch spliting is correct

## Reference
* Bianco, S., Cusano, C., Schettini, R.: Single and multiple illuminant estimation using convolutional neural networks. arXiv preprint (2015).
* Bianco, S., Cusano, C., Schettini, R.: Deep Specialized Network for Illuminant Estimation. arXiv preprint (2015).