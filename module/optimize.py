import tensorflow as tf
from .loss import compute_loss


def build_optimizer(batch_size, data_num):
    boundaries = [data_num // batch_size * epoch for epoch in (1, 50, 60, 70)]
    values = [1e-5, 1e-3, 1e-4, 1e-6, 1e-7]
    learning_rate_fn = tf.optimizers.schedules.PiecewiseConstantDecay(
        boundaries=boundaries, values=values
    )

    optimizer = tf.optimizers.SGD(learning_rate=learning_rate_fn, momentum=0.9)

    return optimizer


@tf.function
def forward_backward(
    input_img,
    true,
    model,
    box_loss_fn,
    clf_loss_fn,
    optimizer,
    num_classes
):
    with tf.GradientTape(persistent=True) as tape:
        pred = model(input_img)
        box_loss, clf_loss, total_loss = compute_loss(true, pred, box_loss_fn, clf_loss_fn, num_classes)

    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return box_loss, clf_loss, total_loss
    