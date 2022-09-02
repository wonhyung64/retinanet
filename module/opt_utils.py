import tensorflow as tf
from .loss_utils import calculate_loss


def build_optimizer(batch_size, data_num):
    boundaries = [data_num // batch_size * epoch for epoch in (1, 50, 60, 70)]
    values = [1e-4, 1e-3, 1e-4, 1e-5, 1e-6]
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_fn)

    return optimizer


@tf.function
def forward_backward(
    img,
    true,
    model,
    optimizer,
    alpha,
    gamma,
    delta,
    total_labels
):
    with tf.GradientTape(persistent=True) as tape:
        pred = model(img)
        reg_loss, cls_loss, total_loss = calculate_loss(true, pred, alpha, gamma, delta, total_labels)

    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))

    return reg_loss, cls_loss, total_loss
