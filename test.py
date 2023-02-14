#%%
import os
import json
import numpy as np
import tensorflow as tf
import neptune.new as neptune

from tqdm import tqdm
from PIL import ImageDraw

from module.anchor import AnchorBox
from module.bbox import swap_xy
from module.model import build_model, DecodePredictions
from module.dataset import load_dataset, build_dataset
from module.neptune import record_result
from module.optimize import build_optimizer
from module.loss import RetinaNetBoxLoss, RetinaNetClassificationLoss
from module.utils import initialize_process, train, evaluate
from module.variable import NEPTUNE_API_KEY, NEPTUNE_PROJECT
from module.args import build_args


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


#%%
args = build_args()
args.data_dir = "/Users/wonhyung64/data"
args.batch_size = 1

run = neptune.init(
    project=NEPTUNE_PROJECT,
    api_token=NEPTUNE_API_KEY,
    run="MOD2-138"
)

datasets, labels, train_num, valid_num, test_num = load_dataset(args.name, args.data_dir)
train_set, valid_set, test_set = build_dataset(datasets, args.batch_size)

experiment_name = run.get_run_url().split("/")[-1].replace("-", "_")
model_name = NEPTUNE_PROJECT.split("-")[1]
experiment_dir = f"./model_weights/{model_name}"
os.makedirs(experiment_dir, exist_ok=True)
weights_dir = f"{experiment_dir}/{experiment_name}"

run["model"].download(f"{weights_dir}.h5")

colors = tf.random.uniform((labels.num_classes, 4), maxval=256, dtype=tf.int32)

model = build_model(labels.num_classes)
model.load_weights(f"{weights_dir}.h5")
decoder = DecodePredictions(confidence_threshold=0.5)


image_shape = tf.cast([512, 512], dtype=tf.float32)
anchor_boxes = AnchorBox().get_anchors(image_shape[0], image_shape[1])
path = "/Users/wonhyung64/data/diagnosis/retina2"


test_progress = tqdm(range(train_num))
for step in test_progress:
    # for _ in range(30):
    #     next(test_set)
    image, gt_boxes, gt_labels = next(train_set)
    # if step <= 2870: continue

    predictions = model(image)

    box_predictions, cls_predictions = predictions
    boxes = decoder._decode_box_predictions(anchor_boxes[None, ...], box_predictions)
    pred_bboxes = tf.expand_dims(swap_xy(boxes[0]), axis=0)
    hw = tf.cast(tf.tile([512, 512], [2]), dtype=tf.float32)
    pred_bboxes = pred_bboxes / hw
    pred_labels = tf.nn.sigmoid(cls_predictions)

    final_bboxes, unscaled_bboxes, final_scores, final_labels = decoder(image, predictions, 1., [512, 512])

    sample = {
        "gt_boxes": gt_boxes[0].numpy(),
        "gt_labels": gt_labels[0].numpy(),
        "anchor_boxes": anchor_boxes.numpy(),
        "pred_bboxes": pred_bboxes[0].numpy(),
        "pred_labels": pred_labels[0].numpy(),
        "final_bboxes": final_bboxes.numpy(),
        "final_labels": final_labels.numpy(),
        "final_scores": final_scores.numpy()
    }

    #save
    with open(f"{path}/sample{step}.json", "w") as f:
        json.dump(sample, f, cls=NumpyEncoder)
        json.dumps(sample, cls=NumpyEncoder)


with open(f"{path}/sample{step}.json", encoding="UTF-8") as f:
    json_load = json.load(f)
json_load = {k: np.asarray(json_load[k]) for k in json_load.keys()}

# %%
