from local_tpu_resolver import LocalTPUClusterResolver
from models.resnet import resnet_18
import os
import time
import math
import argparse
import config
import numpy as np
from prepare_data import generate_datasets
from models.my_adam import MyAdam

tf.config.set_soft_device_placement(True)
tf.random.set_seed(123)

def record(train_recorder, text):
    if train_recorder:
        train_recorder.write(text)
        train_recorder.flush()

def parse_args():
    desc = "Tensorflow implementation of Resnet"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--model', type=str, default='resnet18', help="The model we use")
    parser.add_argument('--rerun', action='store_true')
    parser.set_defaults(rerun=False)
    parser.add_argument('--record', action='store_true')
    parser.set_defaults(record=True)
    return parser.parse_args()


def get_model(m_name, seed):
    if m_name == 'resnet18':
        model = resnet_18(seed, m_name)
        model.build(input_shape=(None, config.image_height, config.image_width, config.channels))
    return model


def main():
    args = parse_args()
    if args is None:
        exit()

    # TPU settings
    tpu_name = os.getenv('TPU_NAME')
    resolver = LocalTPUClusterResolver()
    tf.tpu.experimental.initialize_tpu_system(resolver)

    strategy = tf.distribute.TPUStrategy(resolver)
    per_replica_batch_size = config.BATCH_SIZE // strategy.num_replicas_in_sync
    print("Finish TPU strategy setting!")


    initial_lr = config.initial_lr[args.model]
    seed = 123

    train_dataset, valid_dataset, train_count, valid_count = generate_datasets(seed)

    train_dataset = strategy.experimental_distribute_dataset(train_dataset)
    valid_dataset = strategy.experimental_distribute_dataset(valid_dataset)
 

    print("Start strategy:")
    with strategy.scope():
        model = get_model(args.model, seed)
        #model.optimizer = tf.keras.optimizers.Adam(learning_rate=initial_lr)
        model.optimizer = MyAdam(learning_rate=initial_lr)
        train_loss = tf.keras.metrics.Mean(name='train_loss')
        train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    
        valid_loss = tf.keras.metrics.Mean(name='valid_loss')
        valid_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='valid_accuracy')

    @tf.function
    def train_step(inputs, rerun):
        def step_fn(inputs):
            images, labels = inputs
            with tf.GradientTape() as tape:
                outputs, _, _, _ = model(images, training=True, inject=False)
                predictions = outputs['logits']
                loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
                avg_loss = tf.nn.compute_average_loss(loss, global_batch_size=config.BATCH_SIZE)

            tvars = model.trainable_variables
            gradients = tape.gradient(avg_loss, tvars)
            if not rerun:
                model.optimizer.apply_gradients(grads_and_vars=list(zip(gradients, tvars)))

            train_loss.update_state(avg_loss * strategy.num_replicas_in_sync)
            train_accuracy.update_state(labels, predictions)
            return avg_loss, gradients
        per_replica_loss, per_replica_gradients = strategy.run(step_fn, args=(inputs,))
        reduced_gradients = strategy.reduce("MEAN", per_replica_gradients, axis=None)
        return per_replica_loss, reduced_gradients


    @tf.function
    def revert(gradients, tvars):
        def step_fn(grads, tvars):
            model.optimizer.reset(grads, tvars)
        return strategy.run(step_fn, args=(gradients,tvars))

    @tf.function
    def valid_step(iterator):
        def step_fn(inputs):
            images, labels = inputs
            outputs , _, _, _ = model(images, training=False)
            predictions = outputs['logits']
            v_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, predictions)
            v_loss = tf.nn.compute_average_loss(v_loss, global_batch_size=config.BATCH_SIZE)

            valid_loss.update_state(v_loss)
            valid_accuracy.update_state(labels, predictions)

        return strategy.run(step_fn, args=(next(iterator),))

    # start training
    steps_per_epoch = math.ceil(train_count / config.BATCH_SIZE)
    valid_steps_per_epoch = math.ceil(valid_count / config.VALID_BATCH_SIZE)

    train_recorder = open('train_recorder_{}.txt'.format('rerun' if args.rerun else 'no_rerun'),'w') if args.record else None
    train_iterator = iter(train_dataset)
    prev_iterator = None
 
 
    total_epochs = config.EPOCHS
    epoch = 0


    start = time.time()
    while epoch < total_epochs:
        train_loss.reset_states()
        train_accuracy.reset_states()
        valid_loss.reset_states()
        valid_accuracy.reset_states()
        step = 0

        train_iterator = iter(train_dataset)
        cur_inputs = None
        prev_inputs = None
        prev_gradients = None
        gradients = None
        while step < steps_per_epoch:
            prev_inputs = cur_inputs
            cur_inputs = next(train_iterator)

            def run_one(dataset_inputs, step, rerun=False):
                train_loss.reset_states()
                train_accuracy.reset_states()
                losses, reduced_gradients = train_step(dataset_inputs, rerun)
                return losses, reduced_gradients

            rerun = args.rerun and step and step % 10 == 0
            prev_gradients = gradients
            _, gradients = run_one(cur_inputs, step, rerun)
            if rerun:
                revert(prev_gradients, model.trainable_variables)
                run_one(prev_inputs, step-1)
                run_one(cur_inputs, step)
            step += 1

        valid_iterator = iter(valid_dataset)
        for _ in range(valid_steps_per_epoch):
            valid_step(valid_iterator)
        epoch += 1
    end = time.time()
    record(train_recorder, "Total elapsed time: {}".format(end - start))


if __name__ == '__main__':
    main()

