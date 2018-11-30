from functools import reduce
import tensorflow as tf

def build_model(is_training, inputs, params):
    images = inputs['images']
    assert images.get_shape().as_list() == [None, params.image_size, params.image_size, 4]

    out = images
    for i, conv_block in enumerate(params.conv_blocks):
        with tf.variable_scope("conv_block_{}".format(i+1)):
            out = tf.layers.conv2d(out, **conv_block["conv"])
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, **conv_block["bn"], training=is_training)
            out = tf.nn.relu(out)
            if "pool" in conv_block:
                out = tf.layers.max_pooling2d(out, **conv_block["pool"])
    out = tf.contrib.layers.flatten(out)
    for i, dense_block in enumerate(params.dense_blocks):
        with tf.variable_scope("dense_block_{}".format(i+1)):
            out = tf.layers.dense(out, **dense_block["dense"])
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, **dense_block["bn"], training=is_training)
            out = tf.nn.relu(out)
    with tf.variable_scope("output_block"):
        logits = tf.layers.dense(out, params.num_labels)

    return logits

def model_fn(mode, inputs, params, reuse=False):
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.int64)

    with tf.variable_scope("model", reuse=reuse):
        logits = build_model(is_training, inputs, params)
        predictions = tf.cast(tf.greater(logits, 0), tf.int64)
    
    loss = tf.losses.sigmoid_cross_entropy(labels, logits)
    true_positives = tf.count_nonzero(predictions * labels)
    true_negatives = tf.count_nonzero((predictions - 1) * (labels - 1))
    false_positives = tf.count_nonzero(predictions * (labels - 1))
    false_negatives = tf.count_nonzero((predictions - 1) * labels)
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    macro_f1_score = 2 * precision * recall / (precision + recall)

    if is_training:
        optimizer = tf.train.AdamOptimizer(params.learning_rate)
        global_step = tf.train.get_or_create_global_step()
        if params.use_batch_norm:
            with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            train_op = optimizer.minimize(loss, global_step=global_step)

    with tf.variable_scope("metrics"):
        metrics = {
            "macro_f1_score": tf.contrib.metrics.f1_score(labels=labels, predictions=tf.cast(tf.greater(logits, 0), tf.int64)),
            "loss": tf.metrics.mean(loss)
        }

    update_metrics_op = tf.group(*[op for _, op in metrics.values()])
    metric_variables = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="metrics")
    metrics_init_op = tf.variables_initializer(metric_variables)

    tf.summary.scalar("loss", loss)
    tf.summary.scalar("macro_f1_score", macro_f1_score)
    tf.summary.image("train_image", inputs["images"])

    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['predictions'] = predictions
    model_spec['loss'] = loss
    model_spec['macro_f1_score'] = macro_f1_score
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
