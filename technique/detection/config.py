# some training parameters
EPOCHS = 5
BATCH_SIZE = 1024
VALID_BATCH_SIZE = 1000
PER_REPLICA_BATCH_SIZE=128
PER_REPLICA_VALID_BATCH_SIZE=125
NUM_CLASSES = 10
image_height = 32
image_width = 32
channels = 3

golden_model_dir = "gs://yiizy/golden_models/cifar10"
inj_log_dir = "/home/yiizy/logs/cifar10"
inj_ckpt_dir = "/home/yiizy/models/cifar10"

