# MTCN
Multi-Task Temporal Convolutional Network 

This is the MTCN model for *[Multi-Task Temporal Convolutional Network for
Predicting Water Quality Sensor Data](https://www.ivivan.com/papers/ICONIP2019.pdf)* accepted by ICONIP 2019.
Paper_online: (https://link.springer.com/chapter/10.1007/978-3-030-36808-1_14)

Data set is not included in this repository. Users can test the model on any time series data.

***

**Important**

**model_hpc.py** is the file used to get the final model for the paper. 

This code is used for training on HPC, and it has some dedicated codes for parallel GPU training.

**model_run.py** is used locally for testing and valication. It has the same model but some parameters may differ from the paper. 

> The paper use the best models I selected from HPC training



Required packages:
* Tensorflow
* tcn
* sklearn
* numpy
* pandas
* matplotlib





