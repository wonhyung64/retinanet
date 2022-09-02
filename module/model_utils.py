import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import (
    Conv2D,
    UpSampling2D,
    ReLU,
    Input,
) 
from typing import List
from .anchor_utils import AnchorBox
from .box_utils import convert_to_corners

class FeatureExtractor(Model):
    def __init__(self, args) -> None:
        super(FeatureExtractor, self).__init__()
        self.shape = args.img_size + [3]
        self.base_model = ResNet50(
            include_top=False,
            input_shape=self.shape,
        )
        self.layer_seq = [
            self.base_model.get_layer(layer).output
            for layer in ["conv3_block4_out", "conv4_block6_out", "conv5_block3_out"]
            ]
        self.backbone = Model(inputs=self.base_model.input, outputs=self.layer_seq)
        self.backbone.trainable = False

        self.conv_c3_1 = Conv2D(256, 1, 1, "same")
        self.conv_c4_1 = Conv2D(256, 1, 1, "same")
        self.conv_c5_1 = Conv2D(256, 1, 1, "same")
        self.conv_c3_3 = Conv2D(256, 3, 1, "same")
        self.conv_c4_3 = Conv2D(256, 3, 1, "same")
        self.conv_c5_3 = Conv2D(256, 3, 1, "same")
        self.conv_c6_3 = Conv2D(256, 3, 2, "same")
        self.conv_c7_3 = Conv2D(256, 3, 2, "same", activation="relu")
        self.upsample = UpSampling2D(2)


    @tf.function
    def call(self, inputs: tf.Tensor) -> List:
        c3, c4, c5 = self.backbone(inputs)
        p3 = self.conv_c3_1(c3)
        p4 = self.conv_c4_1(c4)
        p5 = self.conv_c5_1(c5)
        p4 = p4 + self.upsample(p5)
        p3 = p3 + self.upsample(p4)
        p3 = self.conv_c3_3(p3)
        p4 = self.conv_c4_3(p4)
        p5 = self.conv_c5_3(p5)
        p6 = self.conv_c6_3(c5)
        p7 = self.conv_c7_3(p6)

        return p3, p4, p5, p6, p7


class RetinaNet(Model):
    def __init__(self, args, total_labels) -> None:
        super(RetinaNet, self).__init__()
        self.feature_extractor = FeatureExtractor(args)
        self.total_labels = total_labels
        self.batch_size = args.batch_size
        self.anchor_counts = 9
        self.prior_prob = tf.constant_initializer(-np.log((1 - args.prob_init) / args.prob_init))
        self.kernel_init = tf.initializers.RandomNormal(0.0, 0.01)

        self.head = tf.keras.Sequential([])
        self.head.add(Input(shape=[None, None, 256]))
        for _ in range(4):
            self.head.add(Conv2D(
                256,
                3,
                padding="same",
                kernel_initializer=self.kernel_init
                ))
            self.head.add(ReLU())
        self.reg = Conv2D(
            self.anchor_counts * 4,
            3,
            1,
            padding="same",
            kernel_initializer=self.kernel_init,
            bias_initializer="zeros"
            )
        self.cls = Conv2D(
            self.anchor_counts * self.total_labels,
            3,
            1,
            padding="same",
            kernel_initializer=self.kernel_init,
            bias_initializer=self.prior_prob
            )


    @tf.function
    def call(self, inputs: tf.Tensor, batch_size=None) -> List:
        feature_maps = self.feature_extractor(inputs)
        reg_outputs, cls_outputs = [], [] 
        for feature_map in feature_maps:
            reg_outputs.append(tf.reshape(self.reg(feature_map), [
                self.batch_size if batch_size == None else batch_size, -1, 4
                ]))
            cls_outputs.append(tf.reshape(self.cls(feature_map), [
                self.batch_size if batch_size == None else batch_size, -1, self.total_labels
                ]))
        reg_outputs = tf.concat(reg_outputs, axis=1)
        cls_outputs = tf.concat(cls_outputs, axis=1)

        return reg_outputs, cls_outputs


def build_model(args, total_labels):
    model = RetinaNet(args, total_labels=total_labels)
    input_shape = [None] + args.img_size + [3]
    model.build(input_shape=input_shape)

    return model


class Decoder(tf.keras.layers.Layer):
    def __init__(
        self,
        args,
        total_labels,
        max_total_size=200,
        score_threshold=0.5,
        iou_threshold=0.5,
        **kwargs
    ):
        super(Decoder, self).__init__(**kwargs)
        self.total_labels = total_labels
        self.max_total_size = max_total_size
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self._anchor_box = AnchorBox()
        self.variances = tf.convert_to_tensor(args.variances, dtype=tf.float32)
        self.img_size = args.img_size


    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self.variances
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)

        return boxes_transformed


    @tf.function
    def call(self, box_pred, cls_pred):
        anchor_boxes = self._anchor_box.get_anchors(self.img_size[0], self.img_size[1])
        cls_predictions = tf.nn.sigmoid(cls_pred)
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_pred)

        boxes, scores, labels, _ =  tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_total_size,
            self.max_total_size,
            self.iou_threshold,
            self.score_threshold,
            clip_boxes=False,
        )

        return boxes, scores, labels
