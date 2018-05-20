import tensorflow as tf
import tensorflow.contrib.layers as layers


def _get_inputs(mode: str) -> tf.data.Dataset:
    ds = tf.data.TextLineDataset(
        filenames=['data/californiahousing.csv.gz'],
        compression_type='GZIP')

    ds = ds.skip(1)
    ds.cache()
    if mode == 'TRAIN':
        ds = ds.skip(1024)
        ds = ds.repeat()
    elif mode == 'EVAL':
        ds = ds.take(1024)

    ds = ds.shuffle(4 << 10)

    def _parse_records(records: tf.Tensor) -> (tf.Tensor, tf.Tensor):
        columns = tf.decode_csv(
            records=records,
            record_defaults=[[0]] + ([[0.0]] * 8))

        return tf.stack(columns[1:]), columns[0]

    ds = ds.map(_parse_records)
    ds = ds.batch(512)
    return ds


def _model_fn(features: tf.Tensor, labels: tf.Tensor, mode: str) -> tf.estimator.EstimatorSpec:
    from losses.emd import squared_earth_mover_distance

    is_training = mode == tf.estimator.ModeKeys.TRAIN

    out = features
    for idx, outputs in enumerate([5, 7]):
        out = layers.fully_connected(
            inputs=layers.dropout(out, is_training=is_training, keep_prob=0.9),
            num_outputs=outputs,
            activation_fn=tf.nn.elu,
            biases_initializer=None,
            normalizer_fn=layers.batch_norm,
            normalizer_params={
                'is_training': is_training,
                'center': True,
                'scale': True,
                'renorm': True,
            })

        tf.summary.histogram(f'Activations{idx}', out)

    out = layers.fully_connected(
        inputs=layers.dropout(out, is_training=is_training, keep_prob=0.9),
        num_outputs=7,
        activation_fn=None,
        biases_initializer=tf.initializers.zeros())

    tf.summary.histogram(f'Logits', out)

    emd = squared_earth_mover_distance(
        predictions=out,
        labels=tf.one_hot(labels, depth=7))

    loss = tf.losses.compute_weighted_loss(emd)

    train_op = layers.optimize_loss(
        loss=loss,
        global_step=None,
        learning_rate=1e-2,
        optimizer=tf.train.AdamOptimizer,
        clip_gradients=1.5,
        summaries=layers.OPTIMIZER_SUMMARIES)

    eval_metric_ops = None
    if mode == tf.estimator.ModeKeys.EVAL:
        accuracy = tf.metrics.accuracy(
            labels=labels,
            predictions=tf.argmax(out, axis=-1))
        auc = tf.metrics.auc(tf.one_hot(labels, depth=7), tf.nn.softmax(out))
        eval_metric_ops = {
            'Accuracy': accuracy,
            'AUC': auc,
        }

    return tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=tf.argmax(out),
        loss=loss,
        train_op=train_op,
        eval_metric_ops=eval_metric_ops)


run_config = tf.estimator.RunConfig()
run_config = run_config.replace(
    save_checkpoints_steps=10000,
    save_summary_steps=100,
)

est = tf.estimator.Estimator(
    model_fn=_model_fn,
    model_dir='/tmp/emd',
    config=run_config)

tf.estimator.train_and_evaluate(
    estimator=est,
    train_spec=tf.estimator.TrainSpec(
        input_fn=lambda: _get_inputs('TRAIN'),
        hooks=[]),
    eval_spec=tf.estimator.EvalSpec(
        input_fn=lambda: _get_inputs('EVAL'),
        steps=1,
        start_delay_secs=5,
        throttle_secs=5,
    ))

