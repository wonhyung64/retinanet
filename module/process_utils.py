import os
import time
import tensorflow as tf
import neptune.new as neptune
from tqdm import tqdm
from .neptune_utils import *
from .args_utils import build_args
from .opt_utils import (
    forward_backward,
    build_optimizer,
)
from .model_utils import build_model
from .test_utils import calculate_ap_const
from .draw_utils import draw_output


def initialize_process(NEPTUNE_API_KEY, NEPTUNE_PROJECT):
    args = build_args()
    os.makedirs(f"{args.data_dir}/data_chkr", exist_ok=True)
    run = plugin_neptune(NEPTUNE_API_KEY, NEPTUNE_PROJECT, args)

    experiment_name = run.get_run_url().split("/")[-1].replace("-", "_")
    model_name = NEPTUNE_PROJECT.split("-")[-1]
    experiment_dir = f"./model_weights/{model_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    weights_dir = f"{experiment_dir}/{experiment_name}"

    return args, run, weights_dir


def run_process(
    args,
    labels,
    train_num,
    valid_num,
    test_num,
    run,
    train_set,
    valid_set,
    test_set,
    weights_dir,
):
    model, buffer_model = build_model(args, total_labels=len(labels))
    optimizer = build_optimizer(args.batch_size, train_num)

    train_time = train(
        run,
        args,
        train_num,
        valid_num,
        train_set,
        valid_set,
        labels,
        model,
        buffer_model,
        optimizer,
        weights_dir,
    )
    mean_ap, mean_test_time = test(
        run,
        test_num,
        test_set,
        model,
        weights_dir,
        labels,
        args,
    )
    record_result(run, weights_dir, train_time, mean_ap, mean_test_time)


def train(
    run,
    args,
    train_num,
    valid_num,
    train_set,
    valid_set,
    labels,
    model,
    buffer_model,
    optimizer,
    weights_dir,
):
    best_mean_ap = 0
    start_time = time.time()
    alpha = args.alpha
    gamma = args.gamma
    delta = args.delta
    epochs = args.epochs
    batch_size = args.batch_size
    weights_decay = args.weights_decay
    total_labels = len(labels)
    for epoch in range(epochs):
        epoch_progress = tqdm(range(train_num // batch_size))
        for _ in epoch_progress:
            img, true = next(train_set)

            reg_loss, cls_loss, total_loss = forward_backward(
                img, true, model, buffer_model, optimizer, alpha, gamma, delta, weights_decay, total_labels
                )
            record_train_loss(run, reg_loss, cls_loss, total_loss)

            epoch_progress.set_description(
                "Epoch {}/{} | reg {:.4f}, cls {:.4f}, total {:.4f}".format(
                    epoch + 1,
                    args.epochs,
                    reg_loss.numpy(),
                    cls_loss.numpy(),
                    total_loss.numpy(),
                )
            )
        mean_ap = validation(
            valid_set, valid_num, model, labels, args
        )

        run["validation/mAP"].log(mean_ap.numpy())

        if mean_ap.numpy() > best_mean_ap:
            best_mean_ap = mean_ap.numpy()
            model.save_weights(f"{weights_dir}.h5")

    train_time = time.time() - start_time

    return train_time

from .model_utils import Decoder
def validation(valid_set, valid_num, model, labels, args):
    decode = Decoder(args, total_labels=len(labels))
    aps = []
    validation_progress = tqdm(range(valid_num))
    for _ in validation_progress:
        img, gt_boxes, gt_labels = next(valid_set)
        reg_pred, cls_pred = model(img, 1)
        final_bboxes, final_scores, final_labels = decode(reg_pred, cls_pred)

        ap = calculate_ap_const(
            final_bboxes, final_labels, gt_boxes, gt_labels, len(labels)
        )
        validation_progress.set_description(
            "Validation | Average_Precision {:.4f}".format(ap)
        )
        aps.append(ap)

    mean_ap = tf.reduce_mean(aps)

    return mean_ap


def test(
    run, test_num, test_set, model, weights_dir, labels, args
):
    model.load_weights(f"{weights_dir}.h5")
    decode = Decoder(args, total_labels = len(labels))

    test_times = []
    aps = []
    test_progress = tqdm(range(test_num))
    for step in test_progress:
        img, gt_boxes, gt_labels = next(test_set)
        start_time = time.time()
        test_time = time.time() - start_time
        reg_pred, cls_pred = model(img, 1)
        final_bboxes, final_scores, final_labels = decode(reg_pred, cls_pred)

        ap = calculate_ap_const(
            final_bboxes, final_labels, gt_boxes, gt_labels, len(labels)
        )
        test_progress.set_description("Test | Average_Precision {:.4f}".format(ap))
        aps.append(ap)
        test_times.append(test_time)

        if step <= 20:
            run["outputs/dtn"].log(
                neptune.types.File.as_image(
                    draw_output(
                        img, final_bboxes, labels, final_labels, final_scores
                    )
                )
            )

    mean_ap = tf.reduce_mean(aps)
    mean_test_time = tf.reduce_mean(test_times)

    return mean_ap, mean_test_time
