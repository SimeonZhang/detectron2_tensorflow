import math
import tensorflow as tf

from .functional import flatten
from ..structures import box_list
from ..structures import box_list_ops


def smooth_l1_loss(*, labels, predictions, beta, reduction='none', scope=None):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    Smooth L1 loss is related to Huber loss, which is defined as:
                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise
    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:
     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.
    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.
    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        The loss with the reduction option applied.
    """
    with tf.name_scope(scope, 'smooth_l1_loss', values=[labels, predictions]):
        if beta < 1e-5:
            loss = tf.abs(labels - predictions)
        else:
            n = tf.abs(labels - predictions)
            loss = tf.where(
                n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta
            )

        if reduction == "mean":
            loss = tf.reduce_mean(loss)
        elif reduction == "sum":
            loss = tf.reduce_sum(loss)
    return loss


def sigmoid_focal_loss(
    *,
    predictions,
    targets,
    alpha=-1.0,
    gamma=2.0,
    reduction="none",
    scope=None
):
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        predictions: A float tensor of arbitrary shape. The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    with tf.name_scope(scope, 'sigmoid_focal_loss', values=[targets, predictions]):
        p = tf.nn.sigmoid(predictions)
        ce_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=predictions, labels=targets)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        if reduction == "mean":
            loss = tf.reduce_mean(loss)
        elif reduction == "sum":
            loss = tf.reduce_sum(loss)

        return loss


def dice_loss(*, predictions, targets, reduction='none', scope=None):
    """
    Args:
        predictions: 
        targets: 
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    with tf.name_scope(scope, 'dice_loss', values=[predictions, targets]):
        is_empty = tf.equal(tf.size(predictions), 0)

        def valid_fn(predictions, targets):
            predictions = flatten(predictions)
            targets = flatten(targets)

            a = tf.reduce_sum(predictions * targets, axis=1)
            b = tf.reduce_sum(predictions * predictions, axis=1)
            c = tf.reduce_sum(targets * targets, axis=1)
            d = (2 * a) / (b + c + 1e-5)
            loss = 1 - d
            return loss

        loss = tf.cond(is_empty, lambda: 0., lambda: valid_fn(predictions, targets))

        if reduction == "mean":
            loss = tf.reduce_mean(loss)
        elif reduction == "sum":
            loss = tf.reduce_sum(loss)

        return loss


def iou_loss(
    *,
    predictions,
    targets,
    iou_type='iou',
    weight=None,
    reduction='none',
    scope=None
):
    """
    Intersetion Over Union (IoU) loss which supports five loss
    different IoU computations:

    * IoU
    * Linear IoU
    * gIoU
    * diou
    * ciou

    Args:
            predictions: Nx4 predicted bounding boxes, BxNx4, (y1, x1, y2, x2)
            targets: Nx4 target bounding boxes, BxNx4, (y1, x1, y2, x2)
            scale: num to scale the loss. This is used to balance loss which is important for iou loss.
                   eg: scale = 12 for ciou in original paper.
            iou_type: iou loss type, coulde be one of ['iou', 'linear_iou', 'giou', 'diou', 'ciou']
            weigt: N loss weight for each instance
    """
    with tf.name_scope(scope, 'iou_loss', values=[predictions, targets]):
        pred_boxlist = box_list.BoxList(tf.reshape(predictions, [-1, 4]))
        gt_boxlist = box_list.BoxList(tf.reshape(targets, [-1, 4]))

        iou = box_list_ops.matched_iou(
            pred_boxlist, gt_boxlist, iou_type=iou_type
        )

        if iou_type == 'iou':
            losses = -tf.log(iou)
        elif iou_type == 'linear_iou':
            losses = 1 - iou
        elif iou_type == 'giou':
            losses = 1 - iou
        elif iou_type == 'diou':
            losses = 1 - iou
        elif iou_type == 'ciou':
            losses = 1 - iou
        else:
            raise NotImplementedError

        if weight is not None:
            losses = losses * weight

        if reduction == "mean":
            losses = tf.reduce_mean(losses)
        elif reduction == "sum":
            losses = tf.reduce_sum(losses)

        return losses