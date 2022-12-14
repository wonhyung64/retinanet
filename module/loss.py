import tensorflow as tf
from tensorflow.keras.losses import Loss


class RetinaNetBoxLoss(Loss):

    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

    @tf.function
    def call(self, y_true, y_pred):
        difference = y_true - y_pred
        absolute_difference = tf.abs(difference)
        squared_difference = difference ** 2
        loss = tf.where(
            tf.less(absolute_difference, self._delta),
            0.5 * squared_difference,
            absolute_difference - 0.5,
        )

        return tf.reduce_sum(loss, axis=-1)


class RetinaNetClassificationLoss(Loss):

    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
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
        alpha = tf.where(tf.equal(y_true, 1.0), self._alpha, (1.0 - self._alpha))
        pt = tf.where(tf.equal(y_true, 1.0), probs, 1 - probs)
        loss = alpha * tf.pow(1.0 - pt, self._gamma) * cross_entropy

        return tf.reduce_sum(loss, axis=-1)


class RetinaNetLoss(Loss):

    def __init__(self, num_classes=80, alpha=0.25, gamma=2.0, delta=1.0):
        super(RetinaNetLoss, self).__init__(reduction="auto", name="RetinaNetLoss")
        self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
        self._box_loss = RetinaNetBoxLoss(delta)
        self._num_classes = num_classes

    def call(self, y_true, y_pred):
        y_pred = tf.cast(y_pred, dtype=tf.float32)
        box_labels = y_true[..., :4]
        box_predictions = y_pred[..., :4]
        cls_labels = tf.one_hot(
            tf.cast(y_true[..., 4], dtype=tf.int32),
            depth=self._num_classes,
            dtype=tf.float32,
        )
        cls_predictions = y_pred[..., 4:]
        positive_mask = tf.cast(tf.greater(y_true[..., 4], -1.0), dtype=tf.float32)
        ignore_mask = tf.cast(tf.equal(y_true[..., 4], -2.0), dtype=tf.float32)
        clf_loss = self._clf_loss(cls_labels, cls_predictions)
        box_loss = self._box_loss(box_labels, box_predictions)
        clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
        box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
        normalizer = tf.reduce_sum(positive_mask, axis=-1)
        clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
        box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)

        return box_loss, clf_loss

def compute_loss(y_true, y_pred, box_loss_fn, clf_loss_fn, num_classes=20):
    box_pred, cls_pred = y_pred
    box_true, cls_true = y_true

    cls_labels = tf.one_hot(
        tf.cast(cls_true, dtype=tf.int32),
        depth=num_classes,
        dtype=tf.float32,
    )
    
    positive_mask = tf.cast(tf.greater(cls_true, -1.0), dtype=tf.float32)
    ignore_mask = tf.cast(tf.equal(cls_true, -2.0), dtype=tf.float32)
    clf_loss = clf_loss_fn(cls_labels, cls_pred)
    box_loss = box_loss_fn(box_true, box_pred)
    clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
    box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
    normalizer = tf.reduce_sum(positive_mask, axis=-1)
    clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
    box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
    clf_loss = tf.reduce_mean(clf_loss)
    box_loss = tf.reduce_mean(box_loss)
    total_loss = box_loss + clf_loss

    return box_loss, clf_loss, total_loss
