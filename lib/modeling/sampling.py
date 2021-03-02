import tensorflow as tf

__all__ = ["subsample_labels"]


def subsample_labels(labels, num_samples, positive_fraction, bg_label):
    """
    Return `num_samples` random samples from `labels`, with a fraction of
    positives no larger than `positive_fraction`.
    Args:
        labels (Tensor): (N, ) label vector with values:
            * -1: ignore
            * bg_label: background ("negative") class
            * otherwise: one or more foreground ("positive") classes
        num_samples (int): The total number of labels with value >= 0 to return.
            Values that are not sampled will be filled with -1 (ignore).
        positive_fraction (float): The number of subsampled labels with values > 0
            is `min(num_positives, int(positive_fraction * num_samples))`. The number
            of negatives sampled is `min(num_negatives, num_samples - num_positives_sampled)`.
            In order words, if there are not enough positives, the sample is filled with
            negatives. If there are also not enough negatives, then as many elements are
            sampled as is possible.
        bg_label (int): label index of background ("negative") class.
    Returns:
        pos_idx, neg_idx (Tensor):
            1D indices. The total number of indices is `num_samples` if possible.
            The fraction of positive indices is `positive_fraction` if possible.
    """
    positive = tf.where(
        tf.logical_and(
            tf.not_equal(labels, -1), tf.not_equal(labels, bg_label)
        )
    )[:, 0]
    negative = tf.where(tf.equal(labels, bg_label))[:, 0]

    num_pos = int(num_samples * positive_fraction)
    # protect against not enough positive examples
    num_pos = tf.minimum(tf.shape(positive)[0], num_pos)
    num_neg = num_samples - num_pos
    # protect against not enough negative examples
    num_neg = tf.minimum(tf.shape(negative)[0], num_neg)

    pos_idx = tf.random_shuffle(positive)[:num_pos]
    neg_idx = tf.random_shuffle(negative)[:num_neg]
    return pos_idx, neg_idx
