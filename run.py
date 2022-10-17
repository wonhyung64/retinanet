import tensorflow as tf
from module.model import build_model, DecodePredictions
from module.dataset import load_dataset, build_dataset
from module.neptune import record_result
from module.optimize import build_optimizer
from module.loss import RetinaNetBoxLoss, RetinaNetClassificationLoss
from module.utils import initialize_process, train, evaluate
from module.variable import NEPTUNE_API_KEY, NEPTUNE_PROJECT


def main():
    args, run, weights_dir = initialize_process(NEPTUNE_API_KEY, NEPTUNE_PROJECT)

    datasets, labels, train_num, valid_num, test_num = load_dataset(args.name, args.data_dir)
    train_set, valid_set, test_set = build_dataset(datasets, args.batch_size)
    colors = tf.random.uniform((labels.num_classes, 4), maxval=256, dtype=tf.int32)

    model = build_model(labels.num_classes)
    decoder = DecodePredictions(confidence_threshold=0.5)
    box_loss_fn = RetinaNetBoxLoss(args.delta)
    clf_loss_fn = RetinaNetClassificationLoss(args.alpha, args.gamma)
    optimizer = build_optimizer(args.batch_size, train_num, args.momentum)

    train_time = train(run, args.epochs, args.batch_size,
        train_num, valid_num, train_set, valid_set, labels,
        model, decoder, box_loss_fn, clf_loss_fn, optimizer, weights_dir)

    model.load_weights(f"{weights_dir}.h5")
    mean_ap, mean_evaltime = evaluate(run, test_set, test_num, model, decoder, labels, "test", colors)
    record_result(run, weights_dir, train_time, mean_ap, mean_evaltime)


if __name__ == "__main__":
    main()
