# ISCA50_AE

We provide our fault injection framework used in our study and the evaluation of our technique. The methodology to inject faults and apply our technique to the DNN training program are similar for all workloads. We will open-source the complete fault injection framework for all DNN workloads.

In each fault injection experiment, we pick a random training epoch, a random training step, a random layer (selected from both layers in the forward pass and the backward pass), and a random software fault model, and continue training the workload to observe the outcome. In order to inject faults to the backward pass and also correctly propagate the error effects, we manually implemented the backward pass for each workload, which can be found in the `fault_injection/models` folder.

We have performed 2.9M fault injection experiments to obtain statistical results. In this artifact evaluation, we provide three reproducible examples of fault injections that correspond to three outcomes (Masked, Immediate INFs/NaNs, and SlowDegrade) reported in our paper. We also provide instructions for running more fault injection experiments.

Our technique's evaluation includes both detection and recovery. To measure detection performance, we perform detection operations 10,000 times in each training step and calculate the geomean of the overhead. For recovery performance, we re-execute the two most recent training iterations once for every 10 training iterations and calculate the geomean of the overhead.

## System Requirements
### Hardware dependencies
Our framework runs on Google Cloud TPU VMs.
### Software dependencies
Our framework requires the following tools:

```
Tensorflow 2.6.0
Numpy 1.19.5
Gdown 4.6.4
```

## Installation 

### Step 1. create Google Cloud TPU VM

```
export PROJECT_ID=${PROJECT_ID}
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --zone={TPU_LOCATION} --accelerator-type={TPU_TYPE} --version=v2-alpha
```

```
PROJECT_ID: The Google cloud user ID.
TPU_NAME: A user defined name.
TPU_LOCATION: The cloud region, e.g., us-central1-a.
TPU_TYPE: The type of the cloud TPU, e.g., v2-8.
```

For more details on creating TPU VMs, please check [this page](https://cloud.google.com/tpu/docs/users-guide-tpu-vm).


### Step 2. ssh to the TPU VM:

```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${TPU_LOCATION} --project ${PROJECT_ID}
```

### Step 3. check numpy and tensorflow versions

```
import numpy
numpy.__version__
import tensorflow
tensorflow.__version__
```
Make sure that the version of numpy is 1.19.5, and the version of tensorflow is 2.6.0. If the versions don't match, please install the correct versions.


### Step 4. clone our github repo.
```
git clone  https://github.com/YLab-UChicago/ISCA_AE.git
```

### Step 5. Download checkpoints from Google Drive.

```
pip install gdown 
gdown --folder https://drive.google.com/drive/folders/1HVRFWY7NI5xr5qzR8yNeSKCRVnJNnqFf?usp=sharing
```
The commands above download checkpoints to folder `fault_injection/ISCA_AE_CKPT`. Please keep this folder name. If `gdown` cannot be found, specify the full path where gdown is installed, mostly likely in `\~/.local/bin`.


## Experiment workflow

### Fault injection

The `reproduce_injections.py` file is the top-level program to perform the entire workflow of a fault injection experiment, which takes in one argument `--file`, which specifies the injection configs, e.g., the target training epoch, target training step, target layer, faulty values, etc. The configs of our three examples are provided in the `injections` folder.

For each injection, the program generates an output file named `replay_inj_TARGET_INJECTION.txt` file under the `fault_injection` directory, which records the training loss, training accuracy for each training iteration, and test loss and test accuracy for each epoch. For Example 1, the file will also record when INF/NaN values are observed.

To execute each example, run:

#### Example 1 (takes approximately 5 minutes). 
```
cd fault_injection
python3 reproduce_injections.py --file injections/inj_immediate_infs_nans.csv
```

#### Example 2 (takes approximately 10-15 minutes).
```
cd fault_injection
python3 reproduce_injections.py --file injections/inj_masked.csv
```

#### Example 3 (takes approximately 10-15 minutes).
```
cd fault_injection
python3 reproduce_injections.py --file injections/inj_slow_degrade.csv
```

### Evaluation of our technique
To evaluate the overhead of detection (takes approximately 15 minutes): we first execute the `detection.py` script without the `--check` flag. This trains the workload without detection. Once this step is complete, we execute the `detection.py` script with the `--check` flag, enabling detection and executing detection operations 10,000 times for every training iteration. These two commands generate two files: `train_recorder_no_check.txt` and `train_recorder_check.txt`, which record the total elapsed time of the training process without and with detection, respectively. Finally, we execute the `calc_overhead.py` script to obtain the overhead.

```
cd technique/detection
python3 detection.py
python3 detection.py --check
python3 calc_overhead.py
```

We follow a similar methodology for recovery (takes approximately 15 minutes), and execute the `replay.py` script without and with the `--rerun` flag. These two commands generate two files: `train_recorder_no_rerun.txt` and `train_recorder_rerun.txt`, which record the total elapsed time of the training process without and with re-execution, respectively. Finally, we execute the `calc_overhead.py` script to obtain the overhead.

```
cd technique/replay
python3 replay.py
python3 replay.py --rerun
python3 calc_overhead.py
```

## Evaluation and expected results

### Fault injection
Once each experiment is done, the output file should contain the following information:

#### Example 1.
NaN values are reported in the same training iteration (epoch 19, iteration 38) where the fault is injected.

A sample expected output file for this example can be found in `fault_injection/expected_results/replay_inj_immediate_infs_nans.txt`.
    
#### Example 2.
After the fault is injected, both the training and test accuracy keeps improving as the training process continues. The final training accuracy is within ~2% compared with the fault-free run.

A sample expected output file for this example can be found in `fault_injection/expected_results/replay_inj_masked.txt`.
A sample fault-free output file can be found in `fault_injection/expected_results/fault_free.txt`.
    
    
#### Example 3.
After the fault is injected, the training accuracy degrades, then stays at a low level through. The final accuracy is significantly lower than that of the error-free runs.

A sample expected output file for this example can be found in `fault_injection/expected_results/replay_inj_slow_degrade.txt`.

### Evaluation of our technique
The command `python calc_overhead.py` prints out the overheads for detection and recovery. Please note that due to the varying throughputs of different versions of TPUs, the overhead may vary slightly. However, as reported in our paper, we expect the overhead for detection to be less than 0.025% and the overhead for recovery to be less than 0.15%. For reference, we provide our `train_recorder_xx.txt` files under the `detection/expected_result` and `replay/expected_result` folders. These files report a detection overhead of 0.016%, and a recovery overhead of 0.12%.
    
## Experiment customization
To run other examples, one can modify the three example files under the `injection` folder and specify different training epochs, training steps, target layers, and faulty values. Checkpoints that belong to other epochs can be downloaded through:
```
    gdown --folder https://drive.google.com/drive/folders/1B4ptjedCX4e1PbzZWVe5Ydfe48BvSwzt?usp=sharing
```
The evaluation process is similar to the examples provided.
