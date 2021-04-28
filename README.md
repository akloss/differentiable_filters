# differentiable_filters
TensorFlow code for the paper [How to Train Your Differentiable Filter](https://arxiv.org/abs/2012.14313). 

Implements the Extended Kalman filter, the Unscented Kalman filter, a sampling based Unscented Kalman filter and the Particle filter in a differentiable way under "filters". 

The file example_training_code.run_example.py together with contexts.example_context.py provides a simple reference for how to set up and train a differentiable filter (DF) in TensorFlow 2.4.1. See below for details.

A note on TensorFlow versions: While the example assumes TensorFlow 2 is installed, the provided filtering code is also usable under TensorFlow 1 (we tested with Tensorflow 1.14). 


## Installation:

You can either use 
```
pip3 install .
```
or 
``` 
python3 setup.py install
```

## Run the example

To run the example script with default parameters, type

```
run_df_example --out-dir=[path to where output should be written] 
```
This will create a small dataset for the simulated disc tracking task that has been described in the paper and then train a differentiable EKF with a learned process model, heteroscedastic observation noise and constant process noise on this task.

The script has some more parameters with which you can experiment to determine the behaviour of the DF and the training process. You can get a list by typing
```
run_df_example -h
```

## General usage

To use the differentiable filters in your project, you mainly need to do two things:

1) Create a context class that describes the problem you want to run the DF on. 
This class needs inherit from contexts.BaseContext and implement the interface defined there. You can look at the file contexts.example_context.py for a simple example.

2) Combine the context with the differentiable filtering algorithm you want to use. You can look at the class FilterApplication defined in example_training_code.run_example.py for reference.
In general, the DFs are implemented as instances of tf.keras.layers.AbstractRNNCell. This means they define how to perform one step of filtering. To apply the DF to a complete sequence, the filter RNN cell has to be wrapped in an RNN layer as you can see in this example
```
# Instantiate a context object
context = ExampleContext(batch_size, filter_type, loss, hetero_q, hetero_r, learned_process)
# Instantiate the differentiable filter cell
filter_cell = ekf.EKFCell(context, 'simple_example')
# Wrap it in an RNN layer
rnn_layer = tf.keras.layers.RNN(filter_cell, return_sequences=True, unroll=False)
```

To run the RNN, we need to provide an initial state for the RNN and the sequence of observations and control inputs. If the belief of the DF is gaussian (as for the EKF), this could look like this
```
# define the initial state consisting of the mean and covariance of the initial belief and a step counter
init_state = (initial_mean,
              tf.reshape(intial_covariance, [self.batch_size, -1]),
              tf.zeros([self.batch_size, 1]))
# define the inputs to the RNN 
inputs = (observations, control_inputs)
# run the RNN
outputs = rnn_layer(inputs, initial_state=init_state)
```


## Code used for the paper

In addition to the differentiable filters, we provide the full code that was used to run the experiments described in the paper for reference.

Under "contexts", the implementation of the three tasks used in the paper (marked by the "paper_" prefix) with their specific implementations for process and sensor model as well as noise models can be found. "paper_training_code" contains functionality for training and evaluation of the DFs.

The training script provided in paper_training_code can be run with the script `run_df_filter_experiment`. It has a great number of parameters to determine the type and behaviour of the DFs and the training process. You can get a list by typing
```
run_filter_experiment -h
```

Please note: This code was written for TensorFlow 1.14. While it is possible to run it with a TensorFlow 2 installation, the code relies heavily on TensorFlow 1 functionality. This code is thus not intended as an example for how to use differentiable filters. 


## Data

Datsets for the disc tracking task can be generated using the `create_disc_tracking_dataset` script like this:

```
create_disc_tracking_dataset --out-dir=[path/to/data] 
```

For the KITTI task, the raw data has to be downloaded first. We ued the data provided for the paper  "Differentiable Particle Filters: End-to-End Learning with Algorithmic Priors" ([Jonschkowski et al. 2018](https://arxiv.org/pdf/1805.11122.pdf)), which can be downloaded using the setup script [here](https://github.com/tu-rbo/differentiable-particle-filters).
Then, use the script `create_kitti_dataset` to create the data.

Generating the pushing data is more complex and is currently not included in this package. If you are interested, please contact me.


## Author:
Alina Kloss

Copyright(c) 2020 Max Planck Gesellschaft

See LICENSE for license details.
