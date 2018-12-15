import os
import tensorflow as tf

def _parse_function(images_dir, name, label, size):
    rgby_image = []
    for color in ["red", "green", "blue", "yellow"]:
        formatspec = os.path.join(images_dir, "{}_"+color+".png")
        image_string = tf.read_file(images_dir + "/" + name + "_" + color + ".png") 
        image_decoded = tf.squeeze(tf.image.decode_png(image_string, channels=1), axis=-1)
        image = tf.image.convert_image_dtype(image_decoded, tf.float32)
        rgby_image.append(image)
    rgby_image = tf.stack(rgby_image, -1)
    rgby_image = tf.image.resize_images(rgby_image, [size, size])
    return rgby_image, label

def train_preprocess(image, label):
    return image, label

def input_fn(mode, images_dir, names, labels, params):
    num_samples = len(names)
    assert num_samples == len(labels)

    parse_fn = lambda n, l: _parse_function(images_dir, n, l, params.image_size)
    train_fn = lambda n, l: train_preprocess(n, l)

    if mode == "train":
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(names), tf.constant(labels)))
            .shuffle(num_samples)
            .map(parse_fn, num_parallel_calls=params.num_parallel_calls)
            .map(train_fn, num_parallel_calls=params.num_parallel_calls)
            .batch(params.batch_size)
            .prefetch(1)
        )
    else:
        dataset = (tf.data.Dataset.from_tensor_slices((tf.constant(names), tf.constant(labels)))
            .map(parse_fn)
            .batch(params.batch_size)
            .prefetch(1)
        )

    iterator = dataset.make_initializable_iterator()
    images, labels = iterator.get_next()
    iterator_init_op = iterator.initializer

    inputs = {
        "images": images,
        "labels": labels,
        "iterator_init_op": iterator_init_op

    }

    return inputs

