# ISCA50_AE

## System Requirements

Our framework runs on Google Cloud TPU VMs.
Please check [this page](https://cloud.google.com/tpu/docs/users-guide-tpu-vm) for how to create TPU VMs in Google Cloud.

To create a Cloud TPU VM, execute:

```
export PROJECT_ID=${PROJECT_ID}
gcloud alpha compute tpus tpu-vm create ${TPU_NAME} --zone={TPU_LOCATION} --accelerator-type={TPU_TYPE} --version=v2-alpha
```

Then ssh to the TPU VM:

```
gcloud alpha compute tpus tpu-vm ssh ${TPU_NAME} --zone=${TPU_LOCATION} --project ${PROJECT_ID}
```
For Seena:
```
PROJECT_ID=superb-runner-316121
TPU_LOCATION=us-central1-b or us-central1-f
TPU_TYPE=v2-8
```


Our framework requires the following tools:

Tensorflow: v2.6.0

Numpy: v1.19.5

For Seena: if numpy is in a wrong version, execute:
```
sudo pip3 uninstall -y numpy
pip3 install numpy==1.19.5
```

After ssh to TPU VM, download our code at :
```
git clone git@github.com:silvaurus/ISCA50_AE.git
```

## Fault injection to DNN training workloads

This folder implements our fault injection framework for representative DNN training workloads. The methodolgy to inject failures to DNN workloads is similar among workloads. We will open source the fault injection framework for other DNN workloads in the future.

Each time, we pick a random training epoch, a random training step, a random layer (including both layers in the forward pass and the backward pass) and a random software fault model to inject hardware failure, and continue training the workloads to observe the outcomes.

In order to inject failures to the backward pass, and also correctly propagate the error effects, we manually implemented the backward pass for each workloads, which can be found in the `fault_injection/models` folder.

We provide reproducible examples of fault injections that generates unexpected outcomes reported in our paper. These injection examples can be found in the `fault_injection/injections` folder. Each injection example is a csv file which includes the details of the fault injection configs such as the target epoch, step, layer, fault model and faulty values. For example, the file `fault_injection/injections/resnet18/inj_immediate_infs_nans.csv` represents a fault injection example that generates immediate INFs/NaNs for the Resnet18 workload.

### For Seena: trick that track the execution while program is running:
Before executing any command, run `screen` first to enter a screen.

Then run the `python3 reproduce_injections.py` program (see below).

Then use Ctrl-A then Ctrl-D to go back to bash. You can check the output file or do anything.

Use `screen -r` to go back to the screen that runs the program.


Step 1. Download existing [checkpoints](https://drive.google.com/drive/folders/1HVRFWY7NI5xr5qzR8yNeSKCRVnJNnqFf?usp=sharing), and place them under `fault_injection/ISCA_AE_CKPT`. We use these checkpoints as each fault injection experiment resume the model from the beginning of the target epoch.

To download the checkpoints in TPU VM comamnd line, use gdown in Google cloud command line API:

```
cd fault_injection
pip install gdown
gdown --folder https://drive.google.com/drive/folders/1HVRFWY7NI5xr5qzR8yNeSKCRVnJNnqFf?usp=sharing
mv ISCA_AE_CKPT/* .
```

Step 2. Run example injections:

```
  cd fault_injection
  python3 reproduce_injections.py --file injections/TARGET_INJECTION
```

For Seena, we only need to run resnet18, so execute:
```
cd fault_injection
python3 reproduce_injections.py --file injections/resnet18/TARGET_INJECTION
```

This command generate a `replay_inj_TARGET_INJECTION.txt` file in the `fault_injection` folder, which records the losses, training accuracy and test accuracy for each training iteration / epoch. It will also records when INF/NaN values will be observed.

The exepected execution outcomes for each example are provided in the `fault_injection/expected_results` folder. Notice that, due to the randomness exists in training workloads, each time the result might varies a little bit. But the unexpected outcomes should still be exhibited.

Users can also specify their own injection csv files to inject arbitrary failures to the workloads.


### For Seena: no need to run below
## Implementation and evaluation of our technique
This folder implements our light-weight hardware failure detection and mitigation techinque.
The detection technique is implemented in folder `technique/detection`.
To run the technique, execute:

```
  cd technique/detection
  python3 detection.py
```

This will run our technique that compares history values in optimizers, or check the moving variance values in normalization layers. This command generates a `train_recorder.txt` file in folder `technique/detection`, which records the checking results for every training iteration.

The mitigation technique is implemented in folder `technique/replay`.
To run the technique, execute:

```
  cd technique/replay
  python3 replay.py
```

This will run our technique that re-execute the most recent two training iterations when the re-execution signal is enabled. This command generates a `train_recorder.txt` file in folder `technique/detection`, which records the loss and accuracy for each training iteration and also demonstrate the progress of re-execution. To demonstrate our technique, in this implementation, re-execution is enabled for every 20 iterations. In real applications, re-execution will only be triggered when we detect a hardware failure, using our detection technique. Notice that, due to the randomness in training workloads, re-execution of previous iterations do not necessarily generate the exact same loss or accuracy values.

