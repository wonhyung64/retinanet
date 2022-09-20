#%%
import os
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds

from module.target import LabelEncoder
from module.model import get_backbone, RetinaNet, DecodePredictions
from module.loss import RetinaNetLoss
from module.preprocess import preprocess_train, preprocess_test
from module.utils import visualize_detections


def build_model(total_labels):
    resnet50_backbone = get_backbone()
    model = RetinaNet(total_labels, resnet50_backbone)

    return model


def build_optimizer(batch_size, data_num):
    # boundaries = [data_num // batch_size * epoch for epoch in (1, 50, 60, 70)]
    # values = [1e-5, 1e-3, 1e-4, 1e-6, 1e-7]
    # learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
    #     boundaries=boundaries, values=values
    # )
    learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
    learning_rate_boundaries = [125, 250, 500, 240000, 360000]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=learning_rate_boundaries, values=learning_rates
    )

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)

    return optimizer


def build_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default="voc/2007")
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--data-dir", type=str, default="/Users/wonhyung64/data")
    parser.add_argument("--img-size", nargs="+", type=int, default=[512, 512])
    parser.add_argument(
        "--variances", nargs="+", type=float, default=[0.1, 0.1, 0.2, 0.2]
    )
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weights-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--gamma", type=float, default=2.0)
    parser.add_argument("--delta", type=float, default=1.0)
    parser.add_argument("--prob_init", type=float, default=0.01)

    try:
        args = parser.parse_args()
    except:
        args = parser.parse_args([])

    return args


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

    valid_set = valid_set.map(preprocess_train, num_parallel_calls=autotune)
    valid_set = valid_set.padded_batch(
        batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
    )
    valid_set = valid_set.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    valid_set = valid_set.apply(tf.data.experimental.ignore_errors())
    valid_set = valid_set.prefetch(autotune)

    test_set = test_set.map(preprocess_test, num_parallel_calls=autotune)
    test_set = test_set.apply(tf.data.experimental.ignore_errors())
    test_set = test_set.prefetch(autotune)

    train_set = iter(train_set)
    valid_set = iter(valid_set)
    test_set = iter(test_set)

    return train_set, valid_set, test_set

#%%
model_dir = "retinanet/"
os.makedirs(model_dir, exist_ok=True)

args = build_args()
name = args.name
epochs = args.epochs
batch_size = args.batch_size
data_dir = args.data_dir

datasets, labels, train_num, valid_num, test_num = load_dataset(name, data_dir)
train_set, valid_set, test_set = build_dataset(datasets, batch_size)

total_labels = len(labels.names)
model = build_model(total_labels)
loss_fn = RetinaNetLoss(total_labels)

optimizer = build_optimizer(batch_size, data_num=train_num)
model.compile(loss=loss_fn, optimizer=optimizer)

callbacks_list = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, "weights" + "_epoch_{epoch}"),
        monitor="loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )
]

#%%
model.fit(
    train_set,
    validation_data=valid_set,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)

# %%
weights_dir = "./retinanet"

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)
int2str = labels.int2str

image, bbox, class_id, input_image, ratio = next(test_set)
detections = inference_model.predict(input_image)
num_detections = detections.valid_detections[0]
class_names = [
    int2str(int(x)) for x in detections.nmsed_classes[0][:num_detections]
]
visualize_detections(
    image,
    detections.nmsed_boxes[0][:num_detections] / ratio,
    class_names,
    detections.nmsed_scores[0][:num_detections],
)

# %%
