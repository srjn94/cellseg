from functools import reduce
import tensorflow as tf
from tensorflow.python.ops import array_ops

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
            if params.use_dropout:
                out = tf.layers.dropout(out, **conv_block["dropout"], training=is_training)
    out = tf.contrib.layers.flatten(out)
    for i, dense_block in enumerate(params.dense_blocks):
        with tf.variable_scope("dense_block_{}".format(i+1)):
            out = tf.layers.dense(out, **dense_block["dense"])
            if params.use_batch_norm:
                out = tf.layers.batch_normalization(out, **dense_block["bn"], training=is_training)
            if params.use_dropout:
                out = tf.layers.dropout(out, **dense_block["dropout"], training=is_training)
            out = tf.nn.relu(out)
    with tf.variable_scope("output_block"):
        logits = tf.layers.dense(out, params.num_labels)

    return logits

def model_fn(mode, inputs, params, reuse=False):
    is_training = (mode == 'train')
    labels = inputs['labels']
    labels = tf.cast(labels, tf.float32)

    with tf.variable_scope("model", reuse=reuse):
        logits = build_model(is_training, inputs, params)
        predictions = tf.cast(tf.greater(logits, 0), tf.float32)
    
    if "loss" not in params.__dict__ or params.loss["name"] == "sigmoid":
        loss = tf.losses.sigmoid_cross_entropy(labels, logits)
    #source: https://github.com/ailias/Focal-Loss-implement-on-Tensorflow/blob/master/focal_loss.py
    elif params.loss["name"] == "focal":
        sigmoid_p = tf.nn.sigmoid(logits)
        zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
        pos_p_sub = array_ops.where(labels > zeros, labels - sigmoid_p, zeros)
        neg_p_sub = array_ops.where(labels > zeros, zeros, sigmoid_p)
        per_entry_cross_ent = -params.loss["alpha"]*(pos_p_sub**params.loss["gamma"])*tf.log(tf.clip_by_value(sigmoid_p,1e-8,1.0)) \
                              -(1-params.loss["alpha"])*(neg_p_sub**params.loss["gamma"])*tf.log(tf.clip_by_value(1.0-sigmoid_p,1e-8,1.0))
        loss = tf.reduce_sum(per_entry_cross_ent)
    else:
        raise

    if params.use_l2_reg:
        # ignore biases and batch-norm means
        l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if "bias" not in v.name and "beta" not in v.name])
        loss = loss + params.l2_reg_weight * l2_loss


    TP = tf.count_nonzero(predictions * labels, axis=0)
    TN = tf.count_nonzero((predictions - 1) * (labels - 1), axis=0)
    FP = tf.count_nonzero(predictions * (labels - 1), axis=0)
    FN = tf.count_nonzero((predictions - 1) * labels, axis=0)

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * precision * recall / (precision + recall)

    macro_accuracy = tf.reduce_mean(accuracy)
    macro_precision = tf.reduce_mean(precision)
    macro_recall = tf.reduce_mean(recall)
    macro_f1_score = tf.reduce_mean(f1_score)

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
#    tf.summary.image("train_image", inputs["images"])
    model_spec = inputs
    model_spec['variable_init_op'] = tf.global_variables_initializer()
    model_spec['predictions'] = predictions
    model_spec['loss'] = loss
    model_spec['f1_score'] = f1_score
    model_spec['macro_f1_score'] = macro_f1_score
    model_spec['metrics_init_op'] = metrics_init_op
    model_spec['metrics'] = metrics
    model_spec['update_metrics'] = update_metrics_op
    model_spec['summary_op'] = tf.summary.merge_all()

    if is_training:
        model_spec['train_op'] = train_op

    return model_spec
