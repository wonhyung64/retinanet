import tensorflow as tf


class RetinaNetRegLoss(tf.losses.Loss):
    def __init__(self, delta):
        super(RetinaNetRegLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    @tf.function
    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = tf.pow(difference, 2.)
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )
        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClsLoss(tf.losses.Loss):
    def __init__(self, alpha, gamma):
        super(RetinaNetClsLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

    @tf.function
    def call(self, y_true, y_pred):
        cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(
            labels=y_true, logits=y_pred
        )
        probs = tf.nn.sigmoid(y_pred)
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1.0 - probs)
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        # loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy
        loss = alpha * tf.clip_by_value(tf.pow(1.0 - pt, self._gamma), 1e-10, 1.) * cross_entropy
        return tf.reduce_sum(loss, axis=-1)


def calculate_loss(true, pred, alpha, gamma, delta, total_labels):
    _cls_loss = RetinaNetClsLoss(alpha, gamma)
    _reg_loss = RetinaNetRegLoss(delta)

    reg_true, cls_true = true
    reg_pred = tf.cast(pred[0], dtype=tf.float32)
    cls_pred = tf.cast(pred[1], dtype=tf.float32)

    positive_mask = tf.cast(tf.greater(cls_true, -1.0), dtype=tf.float32)
    ignore_mask = tf.cast(tf.equal(cls_true, -2.0), dtype=tf.float32)

    cls_true = tf.one_hot(
        tf.cast(cls_true, dtype=tf.int32),
        depth=total_labels,
        dtype=tf.float32,
    )

    reg_loss = _reg_loss(reg_true, reg_pred)
    cls_loss = _cls_loss(cls_true, cls_pred)

    reg_loss = tf.where(tf.equal(positive_mask, 1.0), reg_loss, 0.0)
    cls_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, cls_loss)

    normalizer = tf.stop_gradient(tf.reduce_sum(positive_mask, axis=-1))
    reg_loss = tf.reduce_mean(tf.math.divide_no_nan(tf.reduce_sum(reg_loss, axis=-1), normalizer))
    cls_loss = tf.reduce_mean(tf.math.divide_no_nan(tf.reduce_sum(cls_loss, axis=-1), normalizer))

    total_loss = cls_loss + reg_loss

    return reg_loss, cls_loss, total_loss
