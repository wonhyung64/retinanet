#%%
import os
# os.chdir('/Users/anseunghwan/Documents/GitHub/archive/retinanet')
# os.chdir(r'D:\archive\retinanet')
os.chdir('/home1/prof/jeon/an/retinanet')

# import re
# import zipfile

import numpy as np
import tensorflow as tf
import tensorflow.keras as K
import tqdm

import matplotlib.pyplot as plt
import tensorflow_datasets as tfds

import datetime
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

from utils import *
from preprocess import *
from anchors import *
from model import *
from labeling import *
#%%
PARAMS = {
    'epochs': 75,
    'batch_size': 2,
    'lr': 1e-3,
    'momentum': 0.9,
    'wd': 1e-4,
    'num_classes': 20,
    
    'alpha': 0.25,
    'gamma': 2.0,
    'delta': 1.0,
    'confidence_threshold': 0.5,
    'prob_init': 0.01,
}
#%%
class_dict = {
    'aeroplane': 1,
    'bicycle': 2,
    'bird': 3,
    'boat': 4,
    'bottle': 5,
    'bus': 6,
    'car': 7,
    'cat': 8,
    'chair': 9,
    'cow': 10,
    'diningtable': 11,
    'dog': 12,
    'horse': 13,
    'motorbike': 14,
    'person': 15,
    'pottedplant': 16,
    'sheep': 17,
    'sofa': 18,
    'train': 19,
    'tvmonitor': 20
}
class_dict = {x:y-1 for x,y in class_dict.items()}
classnum_dict = {y:x for x,y in class_dict.items()}

class DecodePredictions(tf.keras.layers.Layer):
    """A Keras layer that decodes predictions of the RetinaNet model.
    Attributes:
      num_classes: Number of classes in the dataset
      confidence_threshold: Minimum class probability, below which detections
        are pruned.
      nms_iou_threshold: IOU threshold for the NMS operation
      max_detections_per_class: Maximum number of detections to retain per
       class.
      max_detections: Maximum number of detections to retain across all
        classes.
      box_variance: The scaling factors used to scale the bounding box
        predictions.
    """

    def __init__(
        self,
        num_classes=20,
        confidence_threshold=0.5,
        nms_iou_threshold=0.5,
        max_detections_per_class=200,
        max_detections=200,
        box_variance=[0.1, 0.1, 0.2, 0.2],
        **kwargs
    ):
        super(DecodePredictions, self).__init__(**kwargs)
        self.num_classes = num_classes
        self.confidence_threshold = confidence_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detections_per_class = max_detections_per_class
        self.max_detections = max_detections

        self._anchor_box = AnchorBox()
        self._box_variance = tf.convert_to_tensor(box_variance, dtype=tf.float32)

    def _decode_box_predictions(self, anchor_boxes, box_predictions):
        boxes = box_predictions * self._box_variance
        boxes = tf.concat(
            [
                boxes[:, :, :2] * anchor_boxes[:, :, 2:] + anchor_boxes[:, :, :2],
                tf.math.exp(boxes[:, :, 2:]) * anchor_boxes[:, :, 2:],
            ],
            axis=-1,
        )
        boxes_transformed = convert_to_corners(boxes)
        return boxes_transformed

    def call(self, images, box_pred, cls_pred):
        image_shape = tf.cast(tf.shape(images), dtype=tf.float32)
        anchor_boxes = self._anchor_box.get_anchors(image_shape[1], image_shape[2])
        cls_predictions = tf.nn.sigmoid(cls_pred)
        boxes = self._decode_box_predictions(anchor_boxes[None, ...], box_pred)

        return tf.image.combined_non_max_suppression(
            tf.expand_dims(boxes, axis=2),
            cls_predictions,
            self.max_detections_per_class,
            self.max_detections,
            self.nms_iou_threshold,
            self.confidence_threshold,
            clip_boxes=False,
        )
#%%
"""
## Generating detections
"""
def prepare_image(image):
    ratio = 512 / tf.reduce_max(tf.shape(image)[:2])
    image_shape = [512, 512]
    image_shape = tf.cast(image_shape, dtype=tf.float32)
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
    image = tf.keras.applications.resnet.preprocess_input(image)
    return tf.expand_dims(image, axis=0), ratio
#%%
'''dataset'''
train_dataset, val_dataset, test_dataset, ds_info = fetch_dataset(PARAMS['batch_size'])
train_dataset = train_dataset.concatenate(val_dataset)

log_path = f'logs/voc2007'

"""
## Initializing and compiling model
"""
resnet50_backbone = get_backbone()
model = RetinaNet(PARAMS['num_classes'], PARAMS['prob_init'], resnet50_backbone)
model.build(input_shape=[None, 512, 512, 3])
model.summary()
# loss_fn = RetinaNetLoss(num_classes)

buffer_model = RetinaNet(PARAMS['num_classes'], PARAMS['prob_init'], resnet50_backbone)
buffer_model.build(input_shape=(None, 512, 512, 3))
buffer_model.set_weights(model.get_weights()) # weight initialization

# learning_rates = [2.5e-06, 0.000625, 0.00125, 0.0025, 0.00025, 2.5e-05]
# learning_rate_boundaries = [125, 250, 500, 50000, 150000]
# optimizer = K.optimizers.SGD(learning_rate=learning_rates[0], momentum=PARAMS['momentum'])

optimizer = K.optimizers.SGD(learning_rate=PARAMS['lr'], 
                             momentum=PARAMS['momentum'])
learning_rate_boundaries = [50, 60, 70]

label_encoder = LabelEncoder()

train_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/train')
# val_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/val')
# test_writer = tf.summary.create_file_writer(f'{log_path}/{current_time}/test')

# total_length = sum(1 for _ in train_dataset)
iteration = (ds_info.splits["train"].num_examples + ds_info.splits["validation"].num_examples) // PARAMS['batch_size']
#%%
for epoch in range(PARAMS['epochs']):
    loss_avg = tf.keras.metrics.Mean()
    loss_box_avg = tf.keras.metrics.Mean()
    loss_clf_avg = tf.keras.metrics.Mean()
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
    
    train_iter = iter(train_dataset)
    test_iter = iter(test_dataset)
    
    progress_bar = tqdm.tqdm(range(iteration), unit='batch')
    for batch_num in progress_bar:
        
        # '''learning rate schedule'''
        # epoch_ = epoch * iteration + batch_num
        # if epoch_ < learning_rate_boundaries[0]: optimizer.lr = learning_rates[0]
        # elif epoch_ < learning_rate_boundaries[1]: optimizer.lr = learning_rates[1]
        # elif epoch_ < learning_rate_boundaries[2]: optimizer.lr = learning_rates[2]
        # elif epoch_ < learning_rate_boundaries[3]: optimizer.lr = learning_rates[3]
        # elif epoch_ < learning_rate_boundaries[4]: optimizer.lr = learning_rates[4]
        # else: optimizer.lr = learning_rates[5]
        
        '''learning rate schedule'''
        if epoch == 0:
            optimizer.lr = PARAMS['lr'] * 0.01
        elif epoch < learning_rate_boundaries[0]: 
            optimizer.lr = PARAMS['lr']
        elif epoch < learning_rate_boundaries[1]: 
            optimizer.lr = PARAMS['lr'] * 0.1
        elif epoch < learning_rate_boundaries[2]: 
            optimizer.lr = PARAMS['lr'] * (0.1 ** 2)
        else:
            optimizer.lr = PARAMS['lr'] * (0.1 ** 3)

        image, bbox, label = next(train_iter)
        image, bbox_true, cls_true = label_encoder.encode_batch(image, bbox, label)        

        with tf.GradientTape(persistent=True) as tape:
            bbox_pred, cls_pred = model(image)
            loss, box_loss, clf_loss = RetinaNetLoss(bbox_true, cls_true, bbox_pred, cls_pred, PARAMS)
                        
        grads = tape.gradient(loss, model.trainable_variables) 
        optimizer.apply_gradients(zip(grads, model.trainable_variables)) 
        '''decoupled weight decay'''
        weight_decay_decoupled(model, buffer_model, decay_rate=PARAMS['wd'] * optimizer.lr)
        
        loss_avg(loss)
        loss_box_avg(box_loss)
        loss_clf_avg(clf_loss)
        accuracy(cls_true, cls_pred)
        
        progress_bar.set_postfix({
            'EPOCH': f'{epoch:04d}',
            'Loss': f'{loss_avg.result():.4f}',
            'Box': f'{loss_box_avg.result():.4f}',
            'CLS': f'{loss_clf_avg.result():.4f}',
            'Accuracy': f'{accuracy.result():.3%}'
        })
    
    with train_writer.as_default():
        tf.summary.scalar('loss', loss_avg.result(), step=epoch)
        tf.summary.scalar('loss_box', loss_box_avg.result(), step=epoch)
        tf.summary.scalar('loss_clf', loss_clf_avg.result(), step=epoch)
        tf.summary.scalar('accuracy', accuracy.result(), step=epoch)

    # Reset metrics every epoch
    loss_avg.reset_states()
    loss_box_avg.reset_states()
    loss_clf_avg.reset_states()
    accuracy.reset_states()
    
    if epoch % 10 == 0:
        """
        save model
        """
        model_path = f'{log_path}/{current_time}'
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model.save_weights(model_path + '/model_{}_epoch{}.h5'.format(current_time, epoch), save_format="h5")
        
    if epoch % 3 == 0:
        """
        ## Building inference model
        """
        image = K.Input(shape=[512, 512, 3], name="image")
        predictions = model(image, training=False)
        detections = DecodePredictions(confidence_threshold=PARAMS['confidence_threshold'])(image, predictions[0], predictions[1])
        inference_model = K.Model(inputs=image, outputs=detections)
        
        """
        save results
        """
        flag = 0
        results = []
        for sample in test_iter:
            flag += 1
            image = tf.cast(sample[0], dtype=tf.float32)
            input_image, ratio = prepare_image(image[0])
            detections = inference_model.predict(input_image)
            num_detections = detections.valid_detections[0]
            class_names = [
                classnum_dict.get(int(x)) for x in detections.nmsed_classes[0][:num_detections]
            ]
            fig = visualize_detections(
                image[0],
                detections.nmsed_boxes[0][:num_detections] / ratio,
                class_names,
                detections.nmsed_scores[0][:num_detections],
                sample[1].numpy()[0],
                [classnum_dict.get(int(x)) for x in sample[2].numpy()[0]]
            )
            fig.canvas.draw()
            results.append(np.array(fig.canvas.renderer._renderer))
            if flag == 10: break
        
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        for i in range(len(results)):
            axes.flatten()[i].imshow(results[i])
            axes.flatten()[i].axis('off')
        plt.tight_layout()
        plt.savefig('{}/sample_{}.png'.format(model_path, epoch),
                    dpi=200, bbox_inches="tight", pad_inches=0.1)
        # plt.show()
        plt.close()
#%%
"""
save model
"""
model_path = f'{log_path}/{current_time}'
if not os.path.exists(model_path):
    os.makedirs(model_path)
model.save_weights(model_path + '/model_{}.h5'.format(current_time), save_format="h5")

with open(model_path + '/args_{}.txt'.format(current_time), "w") as f:
    for key, value, in PARAMS.items():
        f.write(str(key) + ' : ' + str(value) + '\n')
#%%
"""
## Building inference model
"""
image = tf.keras.Input(shape=[512, 512, 3], name="image")
predictions = model(image, training=False)
detections = DecodePredictions(confidence_threshold=PARAMS['confidence_threshold'])(image, predictions[0], predictions[1])
inference_model = tf.keras.Model(inputs=image, outputs=detections)
#%%
"""
save results
"""
test_iter = iter(test_dataset)
flag = 0
results = []
for sample in test_iter:
    flag += 1
    image = tf.cast(sample[0], dtype=tf.float32)
    input_image, ratio = prepare_image(image[0])
    detections = inference_model.predict(input_image)
    num_detections = detections.valid_detections[0]
    class_names = [
        classnum_dict.get(int(x)) for x in detections.nmsed_classes[0][:num_detections]
    ]
    fig = visualize_detections(
        image[0],
        detections.nmsed_boxes[0][:num_detections] / ratio,
        class_names,
        detections.nmsed_scores[0][:num_detections],
        sample[1].numpy()[0],
        [classnum_dict.get(int(x)) for x in sample[2].numpy()[0]]
    )
    fig.canvas.draw()
    results.append(np.array(fig.canvas.renderer._renderer))
    if flag == 10: break

fig, axes = plt.subplots(2, 5, figsize=(25, 10))
for i in range(len(results)):
    axes.flatten()[i].imshow(results[i])
    axes.flatten()[i].axis('off')
plt.tight_layout()
plt.savefig('{}/sample_{}.png'.format(model_path, epoch),
            dpi=200, bbox_inches="tight", pad_inches=0.1)
# plt.show()
plt.close()
#%%
"""
mAP
"""
#%%
def generate_iou(anchors, gt_boxes):
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = tf.split(anchors, 4, axis=-1) 
    gt_y1, gt_x1, gt_y2, gt_x2 = tf.split(gt_boxes, 4, axis=-1) 
    
    bbox_area = tf.squeeze((bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1), axis=-1) 
    gt_area = tf.squeeze((gt_y2 - gt_y1) * (gt_x2 - gt_x1), axis = -1)
    
    x_top = tf.maximum(bbox_x1, tf.transpose(gt_x1, [0, 2, 1])) 
    y_top = tf.maximum(bbox_y1, tf.transpose(gt_y1, [0, 2, 1]))
    x_bottom = tf.minimum(bbox_x2, tf.transpose(gt_x2, [0, 2, 1]))
    y_bottom = tf.minimum(bbox_y2, tf.transpose(gt_y2, [0, 2, 1]))
    
    intersection_area = tf.maximum(x_bottom - x_top, 0) * tf.maximum(y_bottom - y_top, 0)
    
    union_area = (tf.expand_dims(bbox_area, -1) + tf.expand_dims(gt_area, 1) - intersection_area)
    
    return intersection_area / union_area
#%%
def calculate_PR(final_bbox, gt_box, mAP_threshold):
    bbox_num = final_bbox.shape[1]
    gt_num = gt_box.shape[1]

    true_pos = tf.Variable(tf.zeros(bbox_num))
    for i in range(bbox_num):
        bbox = tf.split(final_bbox, bbox_num, axis=1)[i]

        iou = generate_iou(bbox, gt_box)

        best_iou = tf.reduce_max(iou, axis=1)
        pos_num = tf.cast(tf.greater(best_iou, mAP_threshold), dtype=tf.float32)
        if tf.reduce_sum(pos_num) >= 1:
            gt_box = gt_box * tf.expand_dims(tf.cast(1 - pos_num, dtype=tf.float32), axis=-1)
            true_pos = tf.tensor_scatter_nd_update(true_pos, [[i]], [1])
    false_pos = 1. - true_pos
    true_pos = tf.math.cumsum(true_pos)
    false_pos = tf.math.cumsum(false_pos) 

    recall = true_pos / gt_num
    precision = tf.math.divide(true_pos, true_pos + false_pos)
    
    return precision, recall
#%%
def calculate_AP_per_class(recall, precision):
    interp = tf.constant([i/10 for i in range(0, 11)])
    AP = tf.reduce_max([tf.where(interp <= recall[i], precision[i], 0.) for i in range(len(recall))], axis=0)
    AP = tf.reduce_sum(AP) / 11
    return AP
#%%
def calculate_AP(final_bboxes, final_labels, gt_boxes, gt_labels):
    mAP_threshold = 0.5
    AP = []
    for c in range(PARAMS['num_classes']):
        if tf.math.reduce_any(final_labels == c) or tf.math.reduce_any(gt_labels == c):
            final_bbox = tf.expand_dims(final_bboxes[final_labels == c], axis=0)
            gt_box = tf.expand_dims(gt_boxes[gt_labels == c], axis=0)

            if final_bbox.shape[1] == 0 or gt_box.shape[1] == 0: 
                ap = tf.constant(0.)
            else:
                precision, recall = calculate_PR(final_bbox, gt_box, mAP_threshold)
                ap = calculate_AP_per_class(recall, precision)
            AP.append(ap)
    if AP == []: AP = 1.0
    else: AP = tf.reduce_mean(AP)
    return AP
#%%
mAP = []
test_iter = iter(test_dataset)
for sample in tqdm.tqdm(test_iter):
    image = tf.cast(sample[0], dtype=tf.float32)
    input_image, ratio = prepare_image(image[0])
    detections = inference_model.predict(input_image)
    
    final_bboxes = detections.nmsed_boxes
    final_labels = detections.nmsed_classes

    gt_boxes = sample[1].numpy()[0]
    gt_labels = sample[2].numpy()[0]
    
    final_bboxes = tf.stack([swap_xy(x[None, ...]) for x in final_bboxes[0]], axis=1)
    gt_boxes = swap_xy(convert_to_corners(gt_boxes))
    
    mAP.append(calculate_AP(final_bboxes, final_labels, gt_boxes[tf.newaxis, ...], gt_labels[None, ...]))
#%%
print('mAP:', tf.reduce_mean(mAP).numpy()) # 0.3980715
#%%