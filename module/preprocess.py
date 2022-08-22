#%%
import tensorflow as tf
import tensorflow_datasets as tfds

from utils import *
#%%
"""
## Preprocessing data
Preprocessing the images involves two steps:
- Resizing the image: Images are resized such that the shortest size is equal
to 800 px, after resizing if the longest side of the image exceeds 1333 px,
the image is resized such that the longest size is now capped at 1333 px.
- Applying augmentation: Random scale jittering  and random horizontal flipping
are the only augmentations applied to the images.
Along with the images, bounding boxes are rescaled and flipped if required.
"""

def random_flip_horizontal(image, boxes):
    """Flips image and boxes horizontally with 50% chance
    Arguments:
        image: A 3-D tensor of shape `(height, width, channels)` representing an image.
        
        boxes: A tensor with shape `(num_boxes, 4)` representing bounding boxes, having normalized coordinates.
    Returns:
        Randomly flipped image and boxes
    """
    if tf.random.uniform(()) > 0.5:
        image = tf.image.flip_left_right(image)
        boxes = tf.stack(
            [1 - boxes[:, 2], boxes[:, 1], 1 - boxes[:, 0], boxes[:, 3]], axis=-1
        )
    return image, boxes

# def resize_and_pad_image(
#     image, min_side=800.0, max_side=1333.0, jitter=[640, 1024], stride=128.0
# ):
#     """Resizes and pads image while preserving aspect ratio.

#     1. Resizes images so that the shorter side is equal to `min_side`
#     2. If the longer side is greater than `max_side`, then resize the image with longer side equal to `max_side`
#     3. Pad with zeros on right and bottom to make the image shape divisible by `stride`

#     Arguments:
#         image: A 3-D tensor of shape `(height, width, channels)` representing an image.
#         min_side: The shorter side of the image is resized to this value, if `jitter` is set to None.
#         max_side: If the longer side of the image exceeds this value after
#             resizing, the image is resized such that the longer side now equals to
#             this value.
#         jitter: A list of floats containing minimum and maximum size for scale
#             jittering. If available, the shorter side of the image will be
#             resized to a random value in this range.
#         stride: The stride of the smallest feature map in the feature pyramid.
#             Can be calculated using `image_size / feature_map_size`.

#     Returns:
#         image: Resized and padded image.
#         image_shape: Shape of the image before padding.
#         ratio: The scaling factor used to resize the image
#     """
#     image_shape = tf.cast(tf.shape(image)[:2], dtype=tf.float32)
#     if jitter is not None:
#         min_side = tf.random.uniform((), jitter[0], jitter[1], dtype=tf.float32)
#     ratio = min_side / tf.reduce_min(image_shape)
#     if ratio * tf.reduce_max(image_shape) > max_side:
#         ratio = max_side / tf.reduce_max(image_shape)
#     image_shape = ratio * image_shape
#     image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))
#     padded_image_shape = tf.cast(
#         tf.math.ceil(image_shape / stride) * stride, dtype=tf.int32
#     )
#     image = tf.image.pad_to_bounding_box(
#         image, 0, 0, padded_image_shape[0], padded_image_shape[1]
#     )
#     return image, image_shape, ratio

def preprocess_data(sample):
    """Applies preprocessing step to a single sample
    Arguments:
        sample: A dict representing a single training sample.
    Returns:
        image: Resized and padded image with random horizontal flipping applied.
            (Resizes images so that both width and height are fixed size)
        
        bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
            of the format `[x, y, width, height]`.
            
        class_id: An tensor representing the class id of the objects, having
            shape `(num_objects,)`.
    """
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)

    image, bbox = random_flip_horizontal(image, bbox)
    # image, image_shape, _ = resize_and_pad_image(image)
    image_shape = [512, 512]
    image_shape = tf.cast(image_shape, dtype=tf.float32)
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox, class_id
#%%
def preprocess_test_data(sample):
    """Applies preprocessing step to a single sample
    Arguments:
        sample: A dict representing a single training sample.
    Returns:
        image: Resized and padded image with random horizontal flipping applied.
            (Resizes images so that both width and height are fixed size)
        
        bbox: Bounding boxes with the shape `(num_objects, 4)` where each box is
            of the format `[x, y, width, height]`.
            
        class_id: An tensor representing the class id of the objects, having
            shape `(num_objects,)`.
    """
    
    """
    1. remove is_difficult
    2. without augmentation
    """
    image = sample["image"]
    bbox = swap_xy(sample["objects"]["bbox"])
    class_id = tf.cast(sample["objects"]["label"], dtype=tf.int32)
    is_difficult = sample["objects"]['is_difficult']

    image_shape = [512, 512]
    image_shape = tf.cast(image_shape, dtype=tf.float32)
    image = tf.image.resize(image, tf.cast(image_shape, dtype=tf.int32))

    bbox = tf.stack(
        [
            bbox[:, 0] * image_shape[1],
            bbox[:, 1] * image_shape[0],
            bbox[:, 2] * image_shape[1],
            bbox[:, 3] * image_shape[0],
        ],
        axis=-1,
    )
    bbox = convert_to_xywh(bbox)
    return image, bbox[tf.logical_not(is_difficult)], class_id[tf.logical_not(is_difficult)], is_difficult
#%%
"""
## Setting up a `tf.data` pipeline
To ensure that the model is fed with data efficiently we will be using
`tf.data` API to create our input pipeline. The input pipeline
consists for the following major processing steps:
- Apply the preprocessing function to the samples
- Create batches with fixed batch size. Since images in the batch can
have different dimensions, and can also have different number of
objects, we use `padded_batch` to the add the necessary padding to create
rectangular tensors
- Create targets for each sample in the batch using `LabelEncoder`
"""

def fetch_dataset(batch_size):
    """
    ## Import VOC 2007 dataset
    """
    (train, validation, test), ds_info = tfds.load(name='voc/2007', split=['train', 'validation', 'test'], with_info=True)

    autotune = tf.data.AUTOTUNE
    train_dataset = train.map(preprocess_data, num_parallel_calls=autotune)
    train_dataset = train_dataset.shuffle(ds_info.splits["train"].num_examples)
    train_dataset = train_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=False
    )
    # train_dataset = train_dataset.map(
    #     label_encoder.encode_batch, num_parallel_calls=autotune
    # )
    train_dataset = train_dataset.apply(tf.data.experimental.ignore_errors())
    train_dataset = train_dataset.prefetch(autotune)

    val_dataset = validation.map(preprocess_data, num_parallel_calls=autotune)
    val_dataset = val_dataset.shuffle(ds_info.splits["validation"].num_examples)
    '''
    label에 negative sample index로 padding?
    '''
    val_dataset = val_dataset.padded_batch(
        batch_size=batch_size, padding_values=(0.0, 1e-8, -1), drop_remainder=False
    )
    # val_dataset = val_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    val_dataset = val_dataset.apply(tf.data.experimental.ignore_errors())
    val_dataset = val_dataset.prefetch(autotune)

    test_dataset = test.map(preprocess_test_data, num_parallel_calls=autotune)
    '''
    label에 negative sample index로 padding?
    '''
    test_dataset = test_dataset.padded_batch(
        batch_size=1, drop_remainder=False
    )
    # test_dataset = test_dataset.map(label_encoder.encode_batch, num_parallel_calls=autotune)
    test_dataset = test_dataset.apply(tf.data.experimental.ignore_errors())
    test_dataset = test_dataset.prefetch(autotune)

    return train_dataset, val_dataset, test_dataset, ds_info
#%%