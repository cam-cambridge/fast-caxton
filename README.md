# Fast-CAXTON regression model

[[Paper](https://onlinelibrary.wiley.com/doi/epdf/10.1002/aisy.202200153)] [[Code](https://github.com/cam-cambridge/fast-caxton)] [[Blog](https://www.matta.ai/research/quantitative-and-real-time-control-of-flow-rate-using-deep-learning)]

Accompanying code to the publication: _"Quantitative and Real-Time Control of 3D Printing Material Flow Through Deep Learning"_

![10 seconds of example video of a print in the dataset](media/example-video.gif)

## Setup

This repository allows you to easily train a regnet neural network to quantitatively predict with regression the relative state of the material flow rate in real time during printing from a single input image.

First create a Python 3 virtual environment and install the requirements - this should only take a couple of minutes. In the paper we used PyTorch (v1.10.1), Torchvision (v0.11.1), and CUDA (v11.3) in this work. You can use later versions if required and this repo has since been updated. See the complete list of requirements in `requirements.txt`. 

```
virtualenv -p python3 env
source env/bin/activate
pip install -r requirements.txt
```

## Usage

Inside the `src` directory are two sub-directories for our `data` and `model`. We use Pytorch-Lightning (v1.1.4) as a wrapper for both the dataset and datamodule classes and for our model.

```
python train.py
```

