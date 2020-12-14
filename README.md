# differentiable_filters
tensorflow (1.14) code for the paper "How to train your differentiable filter". 

Implements the Extended Kalman filter, the Unscented Kalman filter, a sampling based Unscented Kalman filter and the Particle filter as differentiable recurrent network layers. 

Under "contexts", the three example tasks used in the paper with their specific implementations for process and sensor model as well as noise models can be found.

Look at filter_network.py to see how a context and a differentiable filter (DF) can be combined into an instance of tf.keras.Model.

## Disclaimer: Work in progress
This initial release mainly contains the code as it was used to produce the results in the paper. Upcoming releases will include an easier-to-use minimal example and a port to tensorflow 2. 

## Installation:

You can either use 
```
pip3 install .
```
or 
``` 
python3 setup.py install
```

## Usage

First create a dataset, for example

```
create_toy_dataset --out-dir=[path/to/data] --num-examples=50
```
This will create a (very) small dataset for the simulated disc tracking task named "toy_pn=0.1_d=5_const", i.e. a set with constant process noise (sigma_p = 0.1) and 5 distractor discs. Note that the filesize for the dataset is still going to be rather big (~ 1GB) and saving might take a while. To further reduce the dataset size, reduce the sequence length or the number of examples.
The paper used --num-examples=2000.


Next, create a directory for the output of the training somewhere. Then, run for example
```
run_filter_experiment --name=my_first_df --problem=toy --filter=ekf --data-name-train=toy_pn=0.1_d=5_const --data-dir=[path/to/data] --out-dir=[path/to/output] --pretrain-process=1 --pretrain-observations=0 --scale=100 --hetero-r=1 --hetero-q=0
```
This will train a differentiable EKF with heteroscedastic observation noise (--hetero-r=1) and constant process noise (--hetero-q=0). pretrain-process=1 defines that a pretrained process model should be used. By default, the pretrained process noise will not be used (to do so, set --use-pretrained-covar=1). If no trained model is available, the pretraining will be done automatically. --scale=100 downscales the complete state-space by a factor of 100, which we found empirically useful.


The script has many more parameters to determine the behaviour of the DF and the training process. You can get a list by typing
```
run_filter_experiment -h
```

## Data
Datsets for the disc trackign task can be generated as described above using the ```create_toy_dataset``` script.
For kitti, the raw data has to be downloaded first. We ued the data provided for the paper  "Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors" (Jonschkowski et al. 2018) (https://arxiv.org/pdf/1805.11122.pdf), which can be downloaded using the setup script here: https://github.com/tu-rbo/differentiable-particle-filters.
Then, use the script ```create_kitti_dataset``` to create the data.

Generating the pushing data is more complex and is currently not included in this package. If you are interested, please contact me.

## Author:
Alina Kloss

Copyright(c) 2020 Max Planck Gesellschaft

See LICENSE for license details.
