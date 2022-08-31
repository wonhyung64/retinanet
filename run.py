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

from module.model_utils import FeatureExtractor
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

base_model = FeatureExtractor(args)
input_shape = [None] + args.img_size + [3]
base_model.build(input_shape=input_shape)
data = tf.convert_to_tensor(np.random.random([1,512,512,3]))

a,b,c,d,e = base_model(data)



features = self.fpn(image, training=training)
    N = tf.shape(image)[0]
    cls_outputs = []
    box_outputs = []
    for feature in features:
        box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
        cls_outputs.append(tf.reshape(self.cls_head(feature), [N, -1, self.num_classes]))
    cls_outputs = tf.concat(cls_outputs, axis=1)
    box_outputs = tf.concat(box_outputs, axis=1)