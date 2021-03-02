import tensorflow as tf

from .fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs
from .box_head import ROI_BOX_HEAD_REGISTRY, FastRCNNConvFCHead, build_box_head
from .roi_heads import ROI_HEADS_REGISTRY, StandardROIHeads
from ...layers import Linear, ShapeSpec, flatten
from .relation_module import (
    compute_geometry_embeddings,
    compute_rank_embeddings,
    ObjectRelationModule
)


@ROI_BOX_HEAD_REGISTRY.register
class RelationBoxHead(FastRCNNConvFCHead):

    def __init__(self, cfg, input_shape: ShapeSpec, **kwargs):
        super().__init__(cfg, input_shape, **kwargs)
        self._init_attention(cfg)

    def _init_attention(self, cfg):
        num_fc = cfg.MODEL.ROI_BOX_HEAD.NUM_FC
        fc_dim = cfg.MODEL.ROI_BOX_HEAD.FC_DIM
        num_groups = cfg.MODEL.ROI_BOX_RELATION_HEAD.NUM_GROUPS
        key_dim = cfg.MODEL.ROI_BOX_RELATION_HEAD.KEY_DIM
        self.geometry_embedding_dim = cfg.MODEL.ROI_BOX_RELATION_HEAD.GEOMETRY_EMBEDDING_DIM

        self.relations = []
        with tf.variable_scope(self.scope, auxiliary_name_scope=False):
            for k in range(num_fc):
                self.relations.append(
                    ObjectRelationModule(
                        input_size=fc_dim,
                        embedding_dim= self.geometry_embedding_dim,
                        key_dim=key_dim,
                        num_groups=num_groups,
                        scope="r{}".format(k + 1)
                    )
                )

    def call(self, x, proposals):
        for layer in self.convs:
            x = layer(x)
        if len(self.fcs):
            if x.shape.ndims > 2:
                x = flatten(x)
            for fc, relation in zip(self.fcs, self.relations):
                x = fc(x)
                x = relation(x, proposals)
        return x

@ROI_HEADS_REGISTRY.register()
class RelationRoiHeads(StandardROIHeads):

    def _forward_box(self, features, proposals):
        """
        Forward logic of the box prediction branch.
        Args:
            features (list[Tensor]): #level input features for box prediction
            proposals (SparseBoxList): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".
        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        box_features = self.box_pooler(features, proposals)
        box_features = self.box_head(box_features, proposals)
        pred_class_logits, pred_proposal_deltas = self.box_predictor(box_features)
        del box_features

        outputs = FastRCNNOutputs(
            self.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            self.smooth_l1_beta,
        )
        if self.training:
            return outputs.losses()
        else:
            pred_instances, _ = outputs.inference(
                self.test_score_thresh, self.test_nms_thresh,
                self.test_detections_per_img, self.test_nms_cls_agnostic
            )
            return pred_instances


    
