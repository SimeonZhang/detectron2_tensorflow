import tensorflow as tf

from ..utils import shape_utils

__all__ = ["Matcher"]


class Matcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.
    The matching is determined by the MxN match_quality_matrix, that
    characterizes how well each (ground-truth, prediction)-pair match each
    other. For example, if the elements are boxes, this matrix may contain box
    intersection-over-union overlap values.
    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(self, thresholds, labels, allow_low_quality_matches=False):
        """
        Args:
            thresholds (list): a list of thresholds used to stratify
                predictions into levels.
            labels (list): a list of values to label predictions belonging at
                each level. A label can be one of {-1, 0, 1} signifying
                {ignore, negative class, positive class}, respectively.
            allow_low_quality_matches (bool): if True, produce additional
                matches for predictions with maximum match quality lower
                than high_threshold. See set_low_quality_matches_ for
                more details.
            For example,
                thresholds = [0.3, 0.5]
                labels = [0, -1, 1]
                All predictions with iou < 0.3 will be marked with 0 and
                thus will be considered as false positives while training.
                All predictions with 0.3 <= iou < 0.5 will be marked with -1
                and thus will be ignored.
                All predictions with 0.5 <= iou will be marked with 1 and
                thus will be considered as true positives.
        """
        # Add -inf and +inf to first and last position in thresholds
        thresholds = thresholds[:]
        thresholds.insert(0, -float("inf"))
        thresholds.append(float("inf"))
        assert all(low <= high for (low, high) in zip(thresholds[:-1], thresholds[1:]))
        assert all(l in [-1, 0, 1] for l in labels)
        assert len(labels) == len(thresholds) - 1
        self.thresholds = thresholds
        self.labels = labels
        self.allow_low_quality_matches = allow_low_quality_matches

    def __call__(
        self,
        match_quality_matrix,
        crowd_matrix=None,
        difficult_matrix=None
    ):
        """
        Args:
            match_quality_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N
                predicted elements.
        Returns:
            matches (Tensor[int64]): a vector of length N, where matches[i] is
                a matched ground-truth index in [0, M)
            match_labels (Tensor[int64]): a vector of length N, where
                pred_labels[i] indicates whether a prediction is a true or
                false positive or ignored
        """
        matrix_shape = shape_utils.combined_static_and_dynamic_shape(match_quality_matrix)
        assert len(matrix_shape) == 2
        M, N = matrix_shape

        def get_matches():
            # match_quality_matrix is M (gt) x N (predicted)
            # Max over gt elements (dim 0) to find best gt candidate for
            # each prediction
            matches = tf.argmax(match_quality_matrix, axis=0)
            matched_vals = tf.reduce_max(match_quality_matrix, axis=0)

            matched_label_inds, matched_label_data = [], []
            for (l, low, high) in zip(
                    self.labels, self.thresholds[:-1], self.thresholds[1:]):
                inds = tf.where(
                    tf.logical_and(
                        tf.greater_equal(matched_vals, low),
                        tf.less(matched_vals, high)
                    )
                )[:, 0]
                data = tf.zeros_like(inds) + l
                matched_label_inds.append(tf.cast(inds, tf.int32))
                matched_label_data.append(data)

            if self.allow_low_quality_matches:
                inds = self.get_low_quality_matches_(match_quality_matrix)
                data = tf.ones_like(inds)
                matched_label_inds.append(tf.cast(inds, tf.int32))
                matched_label_data.append(data)

            match_labels = tf.dynamic_stitch(matched_label_inds, matched_label_data)
            return matches, match_labels

        matches, match_labels = tf.cond(
            tf.greater(M, 0),
            get_matches,
            lambda: (
                tf.zeros([N], dtype=tf.int64), tf.zeros([N], dtype=tf.int64)
            )
        )

        if crowd_matrix is not None:
            crowd_bool = tf.cond(
                tf.shape(crowd_matrix)[0] > 0,
                lambda: tf.reduce_max(crowd_matrix, axis=0) > 1e-3,
                lambda: tf.zeros(tf.shape(crowd_matrix)[1], dtype=tf.bool)
            )
            match_labels = tf.where(
                tf.logical_and(match_labels == 0, crowd_bool),
                tf.zeros_like(match_labels) - 1,
                match_labels
            )

        if difficult_matrix is not None:
            difficult_bool = tf.cond(
                tf.shape(difficult_matrix)[0] > 0,
                lambda: tf.reduce_max(difficult_matrix, axis=0) > self.thresholds[1],
                lambda: tf.zeros(tf.shape(difficult_matrix)[1], dtype=tf.bool)
            )
            match_labels = tf.where(
                tf.logical_and(match_labels == 0, difficult_bool),
                tf.zeros_like(match_labels) - 1,
                match_labels
            )
        return matches, match_labels

    def get_low_quality_matches_(self, match_quality_matrix):
        """
        Produce additional matches for predictions that have only low-quality
        matches. Specifically, for each ground-truth G find the set of
        predictions that have maximum overlap with it (including ties);
        for each prediction in that set, if it is unmatched, then match it to
        the ground-truth G.
        This function implements the RPN assignment case (i) in Sec. 3.1.2 of
        the Faster R-CNN paper: https://arxiv.org/pdf/1506.01497v3.pdf.
        """
        # For each gt, find the prediction with which it has highest quality
        highest_quality_foreach_gt = tf.reduce_max(match_quality_matrix, axis=1)
        # Find the highest quality match available, even if it is low,
        # including ties.
        # Note that the matches qualities must be positive due to the use of
        # `torch.nonzero`.
        gt_pred_pairs_of_highest_quality = tf.where(tf.equal(
            match_quality_matrix, highest_quality_foreach_gt[:, None]
        ))
        # Example gt_pred_pairs_of_highest_quality:
        #   tensor([[    0, 39796],
        #           [    1, 32055],
        #           [    1, 32070],
        #           [    2, 39190],
        #           [    2, 40255],
        #           [    3, 40390],
        #           [    3, 41455],
        #           [    4, 45470],
        #           [    5, 45325],
        #           [    5, 46390]])
        # Each row is a (gt index, prediction index)
        # Note how gt items 1, 2, 3, and 5 each have two ties

        pred_inds_to_update = gt_pred_pairs_of_highest_quality[:, 1]
        return pred_inds_to_update


class YOLOMatcher(object):
    """
    This class assigns to each predicted "element" (e.g., a box) a ground-truth
    element. Each predicted element will have exactly zero or one matches; each
    ground-truth element may be matched to zero or more predicted elements.
    The matching is determined by the MxN match_quality_matrix, that
    characterizes how well each (ground-truth, prediction)-pair match each
    other. For example, if the elements are boxes, this matrix may contain box
    intersection-over-union overlap values.
    The matcher returns (a) a vector of length N containing the index of the
    ground-truth element m in [0, M) that matches to prediction n in [0, N).
    (b) a vector of length N containing the labels for each prediction.
    """

    def __init__(self, threshold):
        """
        Args:
            threshold:
        """
        self.threshold = threshold

    def __call__(
        self,
        anchor_matrix,
        pred_matrix,
        crowd_matrix=None,
        difficult_matrix=None
    ):
        """
        Args:
            anchor_matrix (Tensor[float]): an MxN tensor, containing the
                pairwise quality between M ground-truth elements and N
                cell anchors.
            pred_matrix (Tensor[float]): an MxN* tensor, containing the
                pairwise quality between M ground-truth elements and N*
                predicted boxes.
        Returns:
            positive_inds (Tensor[int64]): a vector of length M, where matches[i] is
                a matched anchor index in [0, N*)
            match_labels (Tensor[int64]): a vector of length N, where
                pred_labels[i] indicates whether a prediction is a true or
                false positive or ignored
        """
        pred_matrix_shape = shape_utils.combined_static_and_dynamic_shape(pred_matrix)
        assert len(pred_matrix_shape) == 2
        M, N = pred_matrix_shape

        def get_matches():
            # match_quality_matrix is M (gt) x N (predicted)
            # Max over gt elements (dim 0) to find best gt candidate for
            # each prediction
            max_iou = tf.reduce_max(pred_matrix, axis=0)

            respond_bgd = tf.where(
                tf.less(max_iou, self.threshold), tf.ones([N], tf.int64), tf.zeros([N], tf.int64)
            )

            best_anchor_inds = tf.argmax(anchor_matrix, axis=1, output_type=tf.int64)
            return best_anchor_inds, respond_bgd

        best_anchor_inds, respond_bgd = tf.cond(
            tf.greater(M, 0),
            get_matches,
            lambda: (
                tf.zeros([M], dtype=tf.int64), tf.zeros([N], dtype=tf.int64)
            )
        )

        if crowd_matrix is not None:
            crowd_bool = tf.cond(
                tf.shape(crowd_matrix)[0] > 0,
                lambda: tf.reduce_max(crowd_matrix, axis=0) > 1e-3,
                lambda: tf.zeros(tf.shape(crowd_matrix)[1], dtype=tf.bool)
            )
            respond_bgd = tf.where(
                tf.logical_and(respond_bgd == 0, crowd_bool),
                tf.zeros_like(respond_bgd) - 1,
                respond_bgd
            )

        if difficult_matrix is not None:
            difficult_bool = tf.cond(
                tf.shape(difficult_matrix)[0] > 0,
                lambda: tf.reduce_max(difficult_matrix, axis=0) > self.threshold,
                lambda: tf.zeros(tf.shape(difficult_matrix)[1], dtype=tf.bool)
            )
            respond_bgd = tf.where(
                tf.logical_and(respond_bgd == 0, difficult_bool),
                tf.zeros_like(respond_bgd) - 1,
                respond_bgd
            )
        return best_anchor_inds, respond_bgd
