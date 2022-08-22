#%%
import tensorflow as tf
import tensorflow.keras as K
import numpy as np
#%%
"""
## Building the ResNet50 backbone
RetinaNet uses a ResNet based backbone, using which a feature pyramid network
is constructed. In the example we use ResNet50 as the backbone, and return the
feature maps at strides 8, 16 and 32.
"""
def get_backbone():
    """Builds ResNet50 with pre-trained imagenet weights"""
    backbone = K.applications.ResNet50(
        include_top=False, input_shape=[None, None, 3]
    )
    c3_output, c4_output, c5_output = [
        backbone.get_layer(layer_name).output
        for layer_name in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
    ]
    return K.Model(
        inputs=[backbone.inputs], outputs=[c3_output, c4_output, c5_output]
    )
#%%
"""
## Building Feature Pyramid Network as a custom layer
"""

class FeaturePyramid(K.layers.Layer):
    """Builds the Feature Pyramid with the feature maps from the backbone.
    Attributes:
        num_classes: Number of classes in the dataset.
        backbone: The backbone to build the feature pyramid from.
            Currently supports ResNet50 only.
    """

    def __init__(self, backbone=None, **kwargs):
        super(FeaturePyramid, self).__init__(name="FeaturePyramid", **kwargs)
        self.backbone = backbone if backbone else get_backbone()
        self.backbone.trainable = False
        
        self.conv_c3_1x1 = K.layers.Conv2D(256, 1, 1, "same")
        self.conv_c4_1x1 = K.layers.Conv2D(256, 1, 1, "same")
        self.conv_c5_1x1 = K.layers.Conv2D(256, 1, 1, "same")
        self.conv_c3_3x3 = K.layers.Conv2D(256, 3, 1, "same")
        self.conv_c4_3x3 = K.layers.Conv2D(256, 3, 1, "same")
        self.conv_c5_3x3 = K.layers.Conv2D(256, 3, 1, "same")
        self.conv_c6_3x3 = K.layers.Conv2D(256, 3, 2, "same")
        self.conv_c7_3x3 = K.layers.Conv2D(256, 3, 2, "same")
        self.upsample_2x = K.layers.UpSampling2D(2)

    def call(self, images, training=False):
        c3_output, c4_output, c5_output = self.backbone(images, training=training)
        p3_output = self.conv_c3_1x1(c3_output)
        p4_output = self.conv_c4_1x1(c4_output)
        p5_output = self.conv_c5_1x1(c5_output)
        p4_output = p4_output + self.upsample_2x(p5_output)
        p3_output = p3_output + self.upsample_2x(p4_output)
        p3_output = self.conv_c3_3x3(p3_output)
        p4_output = self.conv_c4_3x3(p4_output)
        p5_output = self.conv_c5_3x3(p5_output)
        p6_output = self.conv_c6_3x3(c5_output)
        p7_output = self.conv_c7_3x3(tf.nn.relu(p6_output))
        return p3_output, p4_output, p5_output, p6_output, p7_output
#%%
# fpn = FeaturePyramid()
# fpn.build(input_shape=[None, 512, 512, 3])
# fpn(tf.random.normal((2, 512, 512, 3)))
#%%
"""
## Building the classification and box regression heads.
The RetinaNet model has separate heads for bounding box regression and
for predicting class probabilities for the objects. These heads are shared
between all the feature maps of the feature pyramid.
"""

def build_head(output_filters, bias_init):
    """Builds the class/box predictions head.
    Arguments:
        output_filters: Number of convolution filters in the final layer.
        bias_init: Bias Initializer for the final convolution layer.
    Returns:
        A keras sequential model representing either the classification
        or the box regression head depending on `output_filters`.
    """
    head = K.Sequential([K.Input(shape=[None, None, 256])])
    kernel_init = tf.initializers.RandomNormal(0.0, 0.01)
    for _ in range(4):
        head.add(
            K.layers.Conv2D(256, 3, padding="same", kernel_initializer=kernel_init)
        )
        head.add(K.layers.ReLU())
    head.add(
        K.layers.Conv2D(
            output_filters,
            3,
            1,
            padding="same",
            kernel_initializer=kernel_init,
            bias_initializer=bias_init,
        )
    )
    return head
#%%
"""
## Building RetinaNet using a subclassed model
"""

class RetinaNet(K.Model):
    """A subclassed Keras model implementing the RetinaNet architecture.
    Attributes:
      num_classes: Number of classes in the dataset.
      backbone: The backbone to build the feature pyramid from.
        Currently supports ResNet50 only.
    """

    def __init__(self, num_classes, prob_init, backbone=None, **kwargs):
        super(RetinaNet, self).__init__(name="RetinaNet", **kwargs)
        self.fpn = FeaturePyramid(backbone)
        self.num_classes = num_classes
        self.prob_init = prob_init

        prior_probability = tf.constant_initializer(-np.log((1 - prob_init) / prob_init))
        self.cls_head = build_head(9 * num_classes, prior_probability)
        self.box_head = build_head(9 * 4, "zeros")

    def call(self, image, training=False):
        features = self.fpn(image, training=training)
        N = tf.shape(image)[0]
        cls_outputs = []
        box_outputs = []
        for feature in features:
            box_outputs.append(tf.reshape(self.box_head(feature), [N, -1, 4]))
            cls_outputs.append(tf.reshape(self.cls_head(feature), [N, -1, self.num_classes]))
        cls_outputs = tf.concat(cls_outputs, axis=1)
        box_outputs = tf.concat(box_outputs, axis=1)
        return box_outputs, cls_outputs
        # return tf.concat([box_outputs, cls_outputs], axis=-1)
#%%
"""
## Implementing Smooth L1 loss and Focal Loss as keras custom losses
"""

class RetinaNetBoxLoss(tf.losses.Loss):
    """Implements Smooth L1 loss"""

    def __init__(self, delta):
        super(RetinaNetBoxLoss, self).__init__(
            reduction="none", name="RetinaNetBoxLoss"
        )
        self._delta = delta

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
#%%
class RetinaNetClassificationLoss(tf.losses.Loss):
    """Implements Focal loss"""

    def __init__(self, alpha, gamma):
        super(RetinaNetClassificationLoss, self).__init__(
            reduction="none", name="RetinaNetClassificationLoss"
        )
        self._alpha = alpha
        self._gamma = gamma

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
#%%
def RetinaNetLoss(bbox_true, cls_true, box_pred, cls_pred, PARAMS):
    num_classes = PARAMS['num_classes']
    alpha = PARAMS['alpha']
    gamma = PARAMS['gamma']
    delta = PARAMS['delta']
    
    _clf_loss = RetinaNetClassificationLoss(alpha, gamma)
    _box_loss = RetinaNetBoxLoss(delta)

    box_pred = tf.cast(box_pred, dtype=tf.float32)
    cls_pred = tf.cast(cls_pred, dtype=tf.float32)
    cls_labels = tf.one_hot(
        tf.cast(cls_true, dtype=tf.int32),
        depth=num_classes,
        dtype=tf.float32,
    )
    positive_mask = tf.cast(tf.greater(cls_true, -1.0), dtype=tf.float32)
    ignore_mask = tf.cast(tf.equal(cls_true, -2.0), dtype=tf.float32)
    '''
    negative sample은 class label이 -1으로 할당되어 있어서 one_hot vector가 모두 0이 되어 loss 계산에 포함되지 않음 (clf_loss = 0)
    '''
    clf_loss = _clf_loss(cls_labels, cls_pred)
    box_loss = _box_loss(bbox_true, box_pred)
    clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
    box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
    normalizer = tf.stop_gradient(tf.reduce_sum(positive_mask, axis=-1))
    clf_loss = tf.reduce_mean(tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer))
    box_loss = tf.reduce_mean(tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer))
    loss = clf_loss + box_loss
    return loss, box_loss, clf_loss
#%%
# class RetinaNetLoss(tf.losses.Loss):
#     """Wrapper to combine both the losses"""

#     def __init__(self, num_classes=20, alpha=0.25, gamma=2.0, delta=1.0):
#         super(RetinaNetLoss, self).__init__(reduction="auto", name="RetinaNetLoss")
#         self._clf_loss = RetinaNetClassificationLoss(alpha, gamma)
#         self._box_loss = RetinaNetBoxLoss(delta)
#         self._num_classes = num_classes

#     def call(self, y_true, y_pred):
#         y_pred = tf.cast(y_pred, dtype=tf.float32)
#         box_labels = y_true[:, :, :4]
#         box_predictions = y_pred[:, :, :4]
#         cls_labels = tf.one_hot(
#             tf.cast(y_true[:, :, 4], dtype=tf.int32),
#             depth=self._num_classes,
#             dtype=tf.float32,
#         )
#         cls_predictions = y_pred[:, :, 4:]
#         positive_mask = tf.cast(tf.greater(y_true[:, :, 4], -1.0), dtype=tf.float32)
#         ignore_mask = tf.cast(tf.equal(y_true[:, :, 4], -2.0), dtype=tf.float32)
#         '''
#         negative sample은 class label이 -1으로 할당되어 있어서 one_hot vector가 모두 0이 되어 loss 계산에 포함되지 않음 (clf_loss = 0)
#         '''
#         clf_loss = self._clf_loss(cls_labels, cls_predictions)
#         box_loss = self._box_loss(box_labels, box_predictions)
#         clf_loss = tf.where(tf.equal(ignore_mask, 1.0), 0.0, clf_loss)
#         box_loss = tf.where(tf.equal(positive_mask, 1.0), box_loss, 0.0)
#         normalizer = tf.reduce_sum(positive_mask, axis=-1)
#         clf_loss = tf.math.divide_no_nan(tf.reduce_sum(clf_loss, axis=-1), normalizer)
#         box_loss = tf.math.divide_no_nan(tf.reduce_sum(box_loss, axis=-1), normalizer)
#         loss = clf_loss + box_loss
#         return loss, box_loss, clf_loss
#%%
# tf.one_hot(
#     tf.cast(-2 * tf.ones((2, 10)), dtype=tf.int32),
#     depth=20,
#     dtype=tf.float32,
# )
#%%