#%%
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#%%
def weight_decay_decoupled(model, buffer_model, decay_rate):
    # weight decay
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        var.assign(var - decay_rate * buffer_var)
    # update buffer model
    for var, buffer_var in zip(model.trainable_weights, buffer_model.trainable_weights):
        buffer_var.assign(var)
#%%
"""
## Implementing utility functions
Bounding boxes can be represented in multiple ways, the most common formats are:
- Storing the coordinates of the corners `[xmin, ymin, xmax, ymax]`
- Storing the coordinates of the center and the box dimensions
`[x, y, width, height]`
Since we require both formats, we will be implementing functions for converting
between the formats.
"""

def swap_xy(boxes):
    """Swaps order the of x and y coordinates of the boxes.
    Arguments:
        boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes.
    Returns:
        swapped boxes with shape same as that of boxes.
    """
    return tf.stack([boxes[:, 1], boxes[:, 0], boxes[:, 3], boxes[:, 2]], axis=-1)

def convert_to_xywh(boxes):
    """Changes the box format to center, width and height.
    Arguments:
        boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[xmin, ymin, xmax, ymax]`.
    Returns:
        converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )

def convert_to_corners(boxes):
    """Changes the box format to corner coordinates
    Arguments:
        boxes: A tensor of rank 2 or higher with a shape of `(..., num_boxes, 4)`
        representing bounding boxes where each box is of the format
        `[x, y, width, height]`.
    Returns:
        converted boxes with shape same as that of boxes.
    """
    return tf.concat(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,
    )
#%%
def visualize_detections(
    image, boxes, classes, scores, gt_boxes, gt_classes, 
    figsize=(7, 7), linewidth=1, color=[0, 0, 1], gt_color=[1, 0, 0]
):
    """Visualize Detections"""
    image = np.array(image, dtype=np.uint8)
    fig = plt.figure(figsize=figsize)
    plt.axis("off")
    plt.imshow(image)
    ax = plt.gca()
    '''prediction'''
    for box, _cls, score in zip(boxes, classes, scores):
        text = "{}: {:.2f}".format(_cls, score)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        patch = plt.Rectangle(
            [x1, y1], w, h, fill=False, edgecolor=color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x1,
            y1,
            text,
            bbox={"facecolor": color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    '''ground-truth'''
    for box, _cls, in zip(gt_boxes, gt_classes):
        text = "{}".format(_cls)
        x, y, w, h = box
        patch = plt.Rectangle(
            [x - w/2, y - h/2], w, h, fill=False, edgecolor=gt_color, linewidth=linewidth
        )
        ax.add_patch(patch)
        ax.text(
            x + w/2,
            y + h/2,
            text,
            bbox={"facecolor": gt_color, "alpha": 0.4},
            clip_box=ax.clipbox,
            clip_on=True,
        )
    # plt.show()
    plt.close()
    return fig
#%%