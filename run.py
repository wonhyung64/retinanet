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
    