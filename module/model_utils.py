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
        self.conv_c7_3 = Conv2D(256, 3, 2, "same", activation="ReLU")
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
        self.anchor_counts = len(args.anchor_ratios) * len(args.anchor_scales)
        self.prior_prob = tf.constant_initializer(-tf.math.log((1 - args.prob_init) / args.prob_init))
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
    def call(self, inputs: tf.Tensor) -> List:
        feature_maps = self.feature_extractor(inputs)
        reg_outputs, cls_outputs = [], [] 
        for feature_map in feature_maps:
            reg_outputs.append(tf.reshape(self.reg(feature_map), [self.batch_size, -1, 4]))
            cls_outputs.append(tf.reshape(self.cls(feature_map), [self.batch_size, -1, self.total_labels]))

        fc1 = self.FC1(inputs)
        fc2 = self.FC2(fc1)
        fc3 = self.FC3(fc2)
        fc4 = self.FC4(fc3)
        fc5 = self.FC5(fc4)
        dtn_reg_output = self.reg(fc5)
        dtn_cls_output = self.cls(fc5)

        return [dtn_reg_output, dtn_cls_output]
