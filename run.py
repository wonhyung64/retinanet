#%%
import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds

from tqdm import tqdm

from module import (
    initialize_process,
    NEPTUNE_API_KEY,
    NEPTUNE_PROJECT,
    load_dataset,
    build_dataset,
)
from module.model_utils import build_model, Decoder
from module.data_utils import *

# %%
if __name__ == "__main__":
    args, run, weights_dir = initialize_process(
        NEPTUNE_API_KEY, NEPTUNE_PROJECT
    )

    datasets, labels, train_num, valid_num, test_num = load_dataset(
        name=args.name, data_dir=args.data_dir
    )
    train_set, valid_set, test_set = build_dataset(
        datasets, args.batch_size, args.img_size
    )
    next(train_set)

    data = tf.convert_to_tensor(np.random.random([4,512,512,3]))

    model = build_model(args, total_labels=20)
    pred_box, pred_cls = model(data)
    decode = Decoder(args, total_labels=20)
    decode(pred_box, pred_cls)
    help(Decoder)

# %%

def build_dataset(datasets, batch_size, img_size):
    train_set, valid_set, test_set = datasets
    data_shapes = ([None, None, None], [None, None], [None])
    padding_values = (
        tf.constant(0, tf.float32),
        tf.constant(0, tf.float32),
        tf.constant(-1, tf.int32),
    )
img_size = args.img_size
    train_set = train_set.map(lambda x: preprocess(x, split="train", img_size=img_size))
    test_set = test_set.map(lambda x: preprocess(x, split="test", img_size=img_size))
    valid_set = valid_set.map(
        lambda x: preprocess(x, split="validation", img_size=img_size)
    )

    train_set = train_set.repeat().padded_batch(
        batch_size,
        padded_shapes=data_shapes,
        padding_values=padding_values,
        drop_remainder=True,
    )
    valid_set = valid_set.repeat().batch(1)
    test_set = test_set.repeat().batch(1)

    train_set = train_set.prefetch(tf.data.experimental.AUTOTUNE)
    valid_set = valid_set.prefetch(tf.data.experimental.AUTOTUNE)
    test_set = test_set.prefetch(tf.data.experimental.AUTOTUNE)

    train_set = iter(train_set)
    valid_set = iter(valid_set)
    test_set = iter(test_set)

    return train_set, valid_set, test_set


def export_data(sample):
    image = Lambda(lambda x: x["image"])(sample)
    gt_boxes = Lambda(lambda x: x["objects"]["bbox"])(sample)
    gt_labels = Lambda(lambda x: x["objects"]["label"])(sample)
    try:
        is_diff = Lambda(lambda x: x["objects"]["is_crowd"])(sample)
    except:
        is_diff = Lambda(lambda x: x["objects"]["is_difficult"])(sample)

    return image, gt_boxes, gt_labels, is_diff


def resize_and_rescale(image, img_size):
    transform = tf.keras.Sequential(
        [
            tf.keras.layers.experimental.preprocessing.Resizing(
                img_size[0], img_size[1]
            ),
            tf.keras.layers.experimental.preprocessing.Rescaling(1.0 / 255.0),
        ]
    )
    image = transform(image)

    return image


def evaluate(gt_boxes, gt_labels, is_diff):
    not_diff = tf.logical_not(is_diff)
    gt_boxes = Lambda(lambda x: x[not_diff])(gt_boxes)
    gt_labels = Lambda(lambda x: x[not_diff])(gt_labels)

    return gt_boxes, gt_labels


def rand_flip_horiz(image: tf.Tensor, gt_boxes: tf.Tensor) -> Tuple:
    if tf.random.uniform([1]) > tf.constant([0.5]):
        image = tf.image.flip_left_right(image)
        gt_boxes = tf.stack(
            [
                Lambda(lambda x: x[..., 0])(gt_boxes),
                Lambda(lambda x: 1.0 - x[..., 3])(gt_boxes),
                Lambda(lambda x: x[..., 2])(gt_boxes),
                Lambda(lambda x: 1.0 - x[..., 1])(gt_boxes),
            ],
            -1,
        )

    return image, gt_boxes


def preprocess(dataset, split, img_size):
    image, gt_boxes, gt_labels, is_diff = export_data(dataset)
    image = resize_and_rescale(image, img_size)
    if split == "train":
        image, gt_boxes = rand_flip_horiz(image, gt_boxes)
    else:
        gt_boxes, gt_labels = evaluate(gt_boxes, gt_labels, is_diff)
    gt_labels = tf.cast(gt_labels, dtype=tf.int32)

    return image, gt_boxes, gt_labels