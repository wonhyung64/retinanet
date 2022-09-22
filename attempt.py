#%%
import os
import tensorflow as tf

from module.model import build_model, DecodePredictions
from module.dataset import load_dataset, build_dataset
from module.optimize import build_optimizer
from module.ap import calculate_ap_const
from module.args import build_args
from module.loss import RetinaNetLoss
from module.draw import draw_output


#%%
args = build_args()
name = args.name
epochs = args.epochs
batch_size = args.batch_size
data_dir = args.data_dir

model_dir = "retinanet/"
os.makedirs(model_dir, exist_ok=True)

datasets, labels, train_num, valid_num, test_num = load_dataset(name, data_dir)
train_set, valid_set, test_set = build_dataset(datasets, batch_size)

model = build_model(labels.num_classes)
loss_fn = RetinaNetLoss(labels.num_classes)

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
model = build_model(labels.num_classes)
latest_checkpoint = tf.train.latest_checkpoint(weights_dir)
model.load_weights(latest_checkpoint)

colors = tf.random.uniform((labels.num_classes, 4), maxval=256, dtype=tf.int32)
decoder = DecodePredictions(confidence_threshold=0.5)

#%%
image, gt_boxes, gt_labels, input_image, ratio = next(test_set)
predictions = model(input_image, training=False)
scaled_bboxes, final_bboxes, final_scores, final_labels = decoder(input_image, predictions, ratio, tf.shape(image)[:2])
ap = calculate_ap_const(scaled_bboxes, final_labels, gt_boxes, gt_labels, labels.num_classes)
draw_output(image, final_bboxes, final_labels, final_scores, labels, colors)


# %%
