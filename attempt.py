#%%
import os
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds
from PIL import ImageDraw

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
    boundaries = [data_num // batch_size * epoch for epoch in (1, 50, 60, 70)]
    values = [1e-5, 1e-3, 1e-4, 1e-6, 1e-7]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries, values=values
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

    # train_set = iter(train_set)
    # valid_set = iter(valid_set)
    test_set = iter(test_set)

    return train_set, valid_set, test_set


def build_detector(weights_dir, total_labels):
    model = build_model(total_labels)
    latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
    model.load_weights(latest_checkpoint)

    input = tf.keras.Input(shape=[None, None, 3], name="image")
    predictions = model(input, training=False)
    detections = DecodePredictions(confidence_threshold=0.5)(input, predictions, ratio)
    inference_model = tf.keras.Model(inputs=input, outputs=detections)

    return inference_model


def draw_output(
    image,
    final_bboxes,
    final_labels,
    final_scores,
    labels,
    colors,
):
    image = tf.keras.preprocessing.image.array_to_img(image)
    draw = ImageDraw.Draw(image)

    idx = final_labels != 0
    final_bboxes = final_bboxes[idx]
    final_labels = final_labels[idx]
    final_scores = final_scores[idx]

    for index, bbox in enumerate(final_bboxes):
        y1, x1, y2, x2 = tf.split(bbox, 4, axis=-1)
        label_index = int(final_labels[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels.names[label_index], final_scores[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    return image

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
model = build_detector(weights_dir, total_labels)


model = build_model(total_labels)
latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)
#%%
colors = tf.random.uniform((labels.num_classes, 4), maxval=256, dtype=tf.int32)
decoder = DecodePredictions(confidence_threshold=0.5)
image, bbox, class_id, input_image, ratio = next(test_set)
predictions = model(input_image, training=False)
final_bboxes, final_scores, final_labels = decoder(input_image, predictions, ratio)
draw_output(image, final_bboxes, final_labels, final_scores, labels, colors)



# %%
