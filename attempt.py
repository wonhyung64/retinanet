#%%
import os
import argparse
import tensorflow as tf
import tensorflow_datasets as tfds

from module.target import LabelEncoder
from module.model import get_backbone, RetinaNet, DecodePredictions
from module.loss import RetinaNetLoss
from module.preprocess import preprocess_data
from module.utils import prepare_image, visualize_detections


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


#%%
args = build_args()
epochs = args.epochs
batch_size = args.batch_size
data_dir = args.data_dir


label_encoder = LabelEncoder()
(train_dataset, val_dataset), dataset_info = tfds.load(
    "voc/2007", split=["train", "validation"], with_info=True, data_dir = f"{data_dir}/tfds"
)

autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.map(preprocess_data, num_parallel_calls=autotune)
train_dataset = train_dataset.padded_batch(
    batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
train_dataset = train_dataset.map(
    label_encoder.encode_batch, num_parallel_calls=autotune
)
train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
train_dataset = train_dataset.prefetch(autotune)

val_dataset = val_dataset.map(preprocess_data, num_parallel_calls=autotune)
val_dataset = val_dataset.padded_batch(
    batch_size=1, padding_values=(0.0, 1e-8, -1), drop_remainder=True
)
val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
val_dataset = val_dataset.prefetch(autotune)

labels = dataset_info.features["objects"]["label"].names
total_labels = len(labels)

model_dir = "retinanet/"
os.makedirs(model_dir, exist_ok=True)

model = build_model(total_labels)
loss_fn = RetinaNetLoss(total_labels)

optimizer = build_optimizer(batch_size, data_num=4500)
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
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks_list,
    verbose=1,
)

# %%
'''
weights_dir = "./retinanet"

latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

image = tf.keras.Input(shape=[None, None, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=0.5)(image, predictions)
inference_model = tf.keras.Model(inputs=image, outputs=detections)


val_dataset = tfds.load("voc/2007", split="validation", data_dir="/Users/wonhyung64/data/tfds")
int2str = dataset_info.features["objects"]["label"].int2str
val_dataset = iter(val_dataset)

sample = next(val_dataset)
image = tf.cast(sample["image"], dtype=tf.float32)
input_image, ratio = prepare_image(image)
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
'''
# %%

