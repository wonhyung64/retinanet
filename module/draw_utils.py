import tensorflow as tf
from PIL import ImageDraw


def draw_output(
    image,
    final_bboxes,
    labels,
    final_labels,
    final_scores,
):
    image = tf.squeeze(image, axis=0)
    image = tf.stack([
            image[..., 2], image[..., 1], image[..., 0]
                
            ], axis=-1)
    image = tf.keras.preprocessing.image.array_to_img(image)
    draw = ImageDraw.Draw(image)

    y1 = final_bboxes[0][..., 0]
    x1 = final_bboxes[0][..., 1]
    y2 = final_bboxes[0][..., 2]
    x2 = final_bboxes[0][..., 3]

    denormalized_box = tf.round(tf.stack([y1, x1, y2, x2], axis=-1))

    colors = tf.random.uniform((len(labels), 4), maxval=256, dtype=tf.int32)

    for index, bbox in enumerate(denormalized_box):
        y1, x1, y2, x2 = tf.split(bbox, 4, axis=-1)

        final_labels_ = tf.reshape(final_labels[0], shape=(200,))
        final_scores_ = tf.reshape(final_scores[0], shape=(200,))
        label_index = int(final_labels_[index])
        color = tuple(colors[label_index].numpy())
        label_text = "{0} {1:0.3f}".format(labels[label_index], final_scores_[index])
        draw.text((x1 + 4, y1 + 2), label_text, fill=color)
        draw.rectangle((x1, y1, x2, y2), outline=color, width=3)

    return image
