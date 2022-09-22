import tensorflow as tf


def build_optimizer(batch_size, data_num):
    boundaries = [data_num // batch_size * epoch for epoch in (1, 50, 60, 70)]
    values = [1e-5, 1e-3, 1e-4, 1e-6, 1e-7]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries, values=values
    )

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)

    return optimizer
