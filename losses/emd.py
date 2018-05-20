import tensorflow as tf


def squared_earth_mover_distance(
        predictions: tf.Tensor,
        labels: tf.Tensor) -> tf.Tensor:
    """
    Squared EMD loss for binned quantiles

    In [6]: squared_earth_mover_distance(
                tf.constant([[.1, .9, 0], [.5, .5, 0], [.9, 0, .1]]),
                tf.constant([1., 0, 0])).eval()
    Out[6]: array([ 0.80999994,  0.25      ,  0.04000002], dtype=float32)
    """

    with tf.name_scope('EarthMoverDistance'):
        return tf.square(tf.reduce_sum(
            tf.cumsum(labels, axis=-1) - tf.cumsum(predictions, axis=-1),
            axis=-1))