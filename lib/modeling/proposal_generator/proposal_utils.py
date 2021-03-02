import tensorflow as tf

from ...structures import box_list_ops
from ...structures import box_list


def add_ground_truth_to_proposals(targets, proposals):
    """
    Call `add_ground_truth_to_proposals_single_image` for all images.
    Args:
        targets(BoxList): list of N elements. Element i is a Boxes
            representing the gound-truth for image i.
        proposals (BoxList): list of N elements. Element i is a Instances
            representing the proposals for image i.
    Returns:
        BoxList: list of N Instances. Each is the proposals for the image,
            with field "proposal_boxes" and "objectness_logits".
    """
    assert targets is not None

    def add_ground_truth_to_proposals_single_image(inputs):
        """
        Augment `proposals` with ground-truth boxes from `gt_boxes`.
        Args:
            Same as `add_ground_truth_to_proposals`, but with gt_boxes and proposals
            per image.
        Returns:
            Same as `add_ground_truth_to_proposals`, but for only one image.
        """
        targets_dict, proposals_dict = inputs
        is_valid_gt_boxes = targets_dict['is_valid']
        target_boxlist = box_list.BoxList.from_tensor_dict(targets_dict)

        image_shape = proposals_dict.pop('image_shape')
        proposal_boxlist = box_list.BoxList.from_tensor_dict(proposals_dict)

        # Concatenating gt_boxes with proposals requires them to have the same fields
        # Assign all ground-truth boxes an objectness logit corresponding to P(object) \approx 1.
        gt_logit_value = tf.log((1.0 - 1e-10) / (1 - (1.0 - 1e-10)))
        pad_logit_value = tf.log(1e-10 / (1 - 1e-10))

        gt_logits = tf.where(is_valid_gt_boxes, [gt_logit_value], [pad_logit_value])
        target_boxlist.add_field('objectness_logits', gt_logits)

        new_proposal_boxlist = box_list_ops.concatenate([proposal_boxlist, target_boxlist])
        new_proposal_dict = new_proposal_boxlist.as_tensor_dict()
        new_proposal_dict['image_shape'] = image_shape
        return new_proposal_dict

    targets_dict = targets.as_tensor_dict()
    proposals_dict = proposals.as_tensor_dict()
    dtype = {
        field: value.dtype for field, value in proposals_dict.items()
    }
    new_proposal_dict = tf.map_fn(
        add_ground_truth_to_proposals_single_image,
        [targets_dict, proposals_dict],
        dtype=dtype,
    )
    return box_list.BoxList.from_tensor_dict(new_proposal_dict, trackings=['image_shape'])
