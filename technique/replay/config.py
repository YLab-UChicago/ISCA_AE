# some training parameters
EPOCHS = 80
BATCH_SIZE = 1024
#BATCH_SIZE = 128
VALID_BATCH_SIZE = 1000
#VALID_BATCH_SIZE = 125
PER_REPLICA_BATCH_SIZE=128
PER_REPLICA_VALID_BATCH_SIZE=125
NUM_CLASSES = 10
image_height = 32
image_width = 32
channels = 3

# TODO: Need to change this
target_accuracy = {
    'resnet18': 0.68,
    'resnet18_nobn': 0.68,
    'resnet18_sgd': 0.68,
    'resnet18_small_decay': 0.68,
    'effnet': 0.691,
    'densenet': 0.68,
    'nfnet': 0.68
    }

initial_lr = {
        'resnet18': 0.001,
        'resnet18_nobn': 0.001,
        'resnet18_sgd': 0.1,
        'resnet18_small_decay': 0.001,
        'effnet': 0.002,
        'densenet': 0.001,
        'nfnet': 0.01
    }

golden_model_dir = "gs://yiizy/golden_models/cifar10"
#golden_model_dir = "/net/scratch/yiizy/golden_models/effnet"
#inj_log_dir = "/net/scratch/yiizy/inject_logs/cifar10/effnet"
inj_log_dir = "/home/yiizy/logs/cifar10"
inj_ckpt_dir = "/home/yiizy/models/cifar10"

