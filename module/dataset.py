import os
import tensorflow as tf
import tensorflow_datasets as tfds
from .target import LabelEncoder
from .preprocess import preprocess_train, preprocess_test


def load_dataset(name, data_dir):
    train1, dataset_info = tfds.load(
        name=name, split="train", data_dir=f"{data_dir}/tfds", with_info=True
    )
    train2 = tfds.load(
        name=name,
        split="validation[100:]",
        data_dir=f"{data_dir}/tfds",
    )
    valid_set = tfds.load(
        name=name,
        split="validation[:100]",
        data_dir=f"{data_dir}/tfds",
    )
    test_set = tfds.load(
        name=name,
        split="train[:10%]",
        data_dir=f"{data_dir}/tfds",
    )
    train_set = train1.concatenate(train2)

    train_num, valid_num, test_num = load_data_num(
        name, data_dir, train_set, valid_set, test_set
    )

    try:
        labels = dataset_info.features["labels"]
    except:
        labels = dataset_info.features["objects"]["label"]

    return (train_set, valid_set, test_set), labels, train_num, valid_num, test_num


def load_data_num(name, data_dir, train_set, valid_set, test_set):
    data_nums = []
    for dataset, dataset_name in (
        (train_set, "train"),
        (valid_set, "validation"),
        (test_set, "test"),
    ):
        data_num_dir = f"{data_dir}/data_chkr/{''.join(char for char in name if char.isalnum())}_{dataset_name}_num.txt"

        if not (os.path.exists(data_num_dir)):
            data_num = build_data_num(dataset, dataset_name)
            with open(data_num_dir, "w") as f:
                f.write(str(data_num))
                f.close()
        else:
            with open(data_num_dir, "r") as f:
                data_num = int(f.readline())
        data_nums.append(data_num)

    return data_nums


def build_data_num(dataset, dataset_name):
    num_chkr = iter(dataset)
    data_num = 0
    print(f"\nCounting number of {dataset_name} data\n")
    while True:
        try:
            next(num_chkr)
        except:
            break
        data_num += 1

    return data_num


def build_dataset(datasets, batch_size):
    autotune = tf.data.AUTOTUNE
    label_encoder = LabelEncoder()
    (train_set, valid_set, test_set) = datasets

    train_set = train_set.map(preprocess_train, num_parallel_calls=autotune)
    train_set = train_set.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    train_set = train_set.map(
        label_encoder.encode_batch, num_parallel_calls=autotune
    )
    train_set = train_set.apply(tf.data.experimental.ignore_errors())
    train_set = train_set.prefetch(autotune)

    valid_set = valid_set.map(preprocess_test, num_parallel_calls=autotune)
    valid_set = valid_set.apply(tf.data.experimental.ignore_errors())
    valid_set = valid_set.prefetch(autotune)

    test_set = test_set.map(preprocess_test, num_parallel_calls=autotune)
    test_set = test_set.apply(tf.data.experimental.ignore_errors())
    test_set = test_set.prefetch(autotune)

    train_set = iter(train_set)
    valid_set = iter(valid_set)
    test_set = iter(test_set)

    return train_set, valid_set, test_set
