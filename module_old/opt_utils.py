import tensorflow as tf
from .loss_utils import calculate_loss


def build_optimizer(batch_size, data_num):
    boundaries = [data_num // batch_size * epoch for epoch in (1, 50, 60, 70)]
    values = [1e-5, 1e-3, 1e-4, 1e-6, 1e-7]
    lr_fn = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr_fn)

    return optimizer


@tf.function
def forward_backward(
    img,
    true,
    model,
    buffer_model,
    optimizer,
    alpha,
    gamma,
    delta,
    weights_decay,
    total_labels
):
    with tf.GradientTape(persistent=True) as tape:
        pred = model(img)
        reg_loss, cls_loss, total_loss = calculate_loss(true, pred, alpha, gamma, delta, total_labels)

    grads = tape.gradient(total_loss, model.trainable_weights)
    optimizer.apply_gradients(zip(grads, model.trainable_weights))
    weight_decay_decoupled(model, buffer_model, weights_decay)

    return reg_loss, cls_loss, total_loss


def weight_decay_decoupled(model, buffer_model, decay_rate):
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        var.assign(var - decay_rate * buffer_var)

    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        buffer_var.assign(var)
