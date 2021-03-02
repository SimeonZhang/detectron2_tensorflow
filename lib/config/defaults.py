from .config import CfgNode as CN

# -----------------------------------------------------------------------------
# Convention about Training / Test specific parameters
# -----------------------------------------------------------------------------
# Whenever an argument can be either used for training or for testing, the
# corresponding name will be post-fixed by a _TRAIN for a training parameter,
# or _TEST for a test-specific parameter.
# For example, the number of images during training will be
# IMAGES_PER_BATCH_TRAIN, while the number of images for testing will be
# IMAGES_PER_BATCH_TEST

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------

_C = CN()

_C.LOGS = CN()
_C.LOGS.ROOT_DIR = ""
_C.LOGS.TRAIN = "train"
_C.LOGS.EVAL = "eval"
_C.LOGS.EXPORT = "export"


_C.SERVING_MODEL = CN()
_C.SERVING_MODEL.FROZEN_GRAPH_FILE_NAME = 'frozen_inference_graph.pb'
_C.SERVING_MODEL.INPUT_OUTPUT_TENSOR_PREFIX = ''
_C.SERVING_MODEL.TYPE = 'Detection'
_C.SERVING_MODEL.LABEL_OFFSET = 1

# -----------------------------------------------------------------------------
# BUILD RECORDS
# -----------------------------------------------------------------------------
_C.BUILD_RECORDS = CN()
_C.BUILD_RECORDS.TYPE = "coco_pano"  # one of "coco_pano", "coco_det"
_C.BUILD_RECORDS.ROOT_DIR = ""  # root dir of the raw data
_C.BUILD_RECORDS.TRAIN_NUM_SHARDS = 16
_C.BUILD_RECORDS.VAL_NUM_SHARDS = 16

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASETS = CN()
_C.DATASETS.ROOT_DIR = ""
_C.DATASETS.TRAIN = "train"
_C.DATASETS.VAL = "val"
_C.DATASETS.CATEGORY_MAP_NAME = 'category_map.json'


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
_C.EVAL = CN()
_C.EVAL.METRICS = ('coco_detection_metrics',)
_C.EVAL.NUM_EVAL = 5000

_C.EVAL.INCLUDE_METRICS_PER_CATEGORY = False
_C.EVAL.ALL_METRICS_PER_CATEGORY = False

_C.EVAL.MAX_EXAMPLE_TO_DRAW = 100
_C.EVAL.MIN_VISUALIZATION_SCORE_THRESH = 0.5

_C.EVAL.PASCAL_MATCHING_IOU_THRESH = 0.5
_C.EVAL.CLASS_AGNOSTIC = False

_C.MODEL = CN()
_C.MODEL.LOAD_PROPOSALS = False
_C.MODEL.MASK_ON = True
_C.MODEL.META_ARCHITECTURE = "GeneralizedRCNN"

# If the WEIGHT starts with a catalog://, like :R-50, the code will look for
# the path in ModelCatalog. Else, it will use it as the specified absolute
# path
_C.PRETRAINS = CN()
_C.PRETRAINS.ROOT = ""
_C.PRETRAINS.DETECTRON2 = ""
_C.PRETRAINS.ONLY_BACKBONE = False
_C.PRETRAINS.BACKBONE = ""
_C.PRETRAINS.WEIGHTS = ""
_C.PRETRAINS.MMDET = ""
_C.PRETRAINS.DARKNET = ""

# -----------------------------------------------------------------------------
# TRANSFORM
# -----------------------------------------------------------------------------
_C.TRANSFORM = CN()

# -----------------------------------------------------------------------------
# RESIZE
# -----------------------------------------------------------------------------
_C.TRANSFORM.RESIZE = CN()
# Size of the smallest side of the image during training
_C.TRANSFORM.RESIZE.MIN_SIZE_TRAIN = (800,)
# Maximum size of the side of the image during training
_C.TRANSFORM.RESIZE.MAX_SIZE_TRAIN = 1333
# Size of the smallest side of the image during testing. Set to zero to disable resize in testing.
_C.TRANSFORM.RESIZE.MIN_SIZE_TEST = 800
# Maximum size of the side of the image during testing
_C.TRANSFORM.RESIZE.MAX_SIZE_TEST = 1333

_C.TRANSFORM.RESIZE.USE_MINI_MASKS = True
_C.TRANSFORM.RESIZE.MINI_MASK_SIZE = 56

_C.MODEL.INPUT_FORMAT = "BGR"
# Values to be used for image normalization (RGB order)
# Default values are the mean pixel value from ImageNet: [123.675, 116.280, 103.530]
_C.MODEL.PIXEL_MEAN = [123.675, 116.280, 103.530]
# When using pre-trained models in Detectron1 or any MSRA models,
# std has been absorbed into its conv1 weights, so the std needs to be set 1.
# Otherwise, you can use [58.395, 57.120, 57.375] (ImageNet std)
_C.MODEL.PIXEL_STD = [1.0, 1.0, 1.0]

# -----------------------------------------------------------------------------
# AUGMENTATION
# -----------------------------------------------------------------------------
_C.AUGMENT = CN()

# -----------------------------------------------------------------
# FLIP / Rotate
# -----------------------------------------------------------------
_C.AUGMENT.HORIZONTAL_FLIP = False
_C.AUGMENT.VERTICAL_FLIP = False
_C.AUGMENT.ROTATE = False
_C.AUGMENT.ROTATE_BOTH_DIRECTION = False
# -----------------------------------------------------------------
# PIXEL VALUE SCALE
# -----------------------------------------------------------------
_C.AUGMENT.PIXEL_VALUE_SCALE = CN({"ENABLED": False})
_C.AUGMENT.PIXEL_VALUE_SCALE.MIN_VALUE = 0.9
_C.AUGMENT.PIXEL_VALUE_SCALE.MAX_VALUE = 1.1

# -----------------------------------------------------------------
# ADJUST BRIGHTNESS
# -----------------------------------------------------------------
_C.AUGMENT.ADJUST_BRIGHTNESS = CN({"ENABLED": False})
_C.AUGMENT.ADJUST_BRIGHTNESS.MAX_DELTA = 0.2

# -----------------------------------------------------------------
# ADJUST CONSTRACT
# -----------------------------------------------------------------
_C.AUGMENT.ADJUST_CONSTRACT = CN({"ENABLED": False})
_C.AUGMENT.ADJUST_CONSTRACT.MIN_DELTA = 0.8
_C.AUGMENT.ADJUST_CONSTRACT.MAX_DELTA = 1.25

# -----------------------------------------------------------------
# ADJUST HUE
# -----------------------------------------------------------------
_C.AUGMENT.ADJUST_HUE = CN({"ENABLED": False})
_C.AUGMENT.ADJUST_HUE.MAX_DELTA = 0.02

# -----------------------------------------------------------------
# ADJUST SATURATION
# -----------------------------------------------------------------
_C.AUGMENT.ADJUST_SATURATION = CN({"ENABLED": False})
_C.AUGMENT.ADJUST_SATURATION.MIN_DELTA = 0.8
_C.AUGMENT.ADJUST_SATURATION.MAX_DELTA = 1.25

# -----------------------------------------------------------------
# DISTORT COLOR
# -----------------------------------------------------------------
_C.AUGMENT.DISTORT_COLOR = CN({"ENABLED": False})
_C.AUGMENT.DISTORT_COLOR.COLOR_ORDERING = 0

# -----------------------------------------------------------------------------
# CROP
# -----------------------------------------------------------------------------
_C.AUGMENT.CROP = CN({"ENABLED": False})
_C.AUGMENT.CROP.MIN_OBJECT_COVERED = 1.0
_C.AUGMENT.CROP.ASPECT_RATIO_RANGE = (0.75, 1.33)
_C.AUGMENT.CROP.AREA_RANGE = (0.1, 1.0)
_C.AUGMENT.CROP.OVERLAP_THRESH = 0.3
_C.AUGMENT.CROP.RANDOM_COEF = 0.0

# -----------------------------------------------------------------------------
# JITTER
# -----------------------------------------------------------------------------
_C.AUGMENT.JITTER_BOX = CN({"ENABLED": False})
_C.AUGMENT.JITTER_BOX.RATIO = 0.05

# _C.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"

# -----------------------------------------------------------------------------
# Segmentation Output
# -----------------------------------------------------------------------------
_C.MODEL.SEGMENTATION_OUTPUT = CN()
_C.MODEL.SEGMENTATION_OUTPUT.FORMAT = "conventional" # one of [raw, fixed, conventional]
_C.MODEL.SEGMENTATION_OUTPUT.FIXED_RESOLUTION = 512

# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
# Number of file shards to read in parallel.
_C.DATALOADER.NUM_READERS = 4
_C.DATALOADER.READ_BLOCK_LENGTH = 1
_C.DATALOADER.FILE_READ_BUFFER_SIZE = 8

_C.DATALOADER.SAMPLE_1_OF_N = 1

_C.DATALOADER.SHUFFLE = True
_C.DATALOADER.FILENAME_SHUFFLE_BUFFER_SIZE = 64
_C.DATALOADER.SHUFFLE_BUFFER_SIZE = 16

_C.DATALOADER.NUM_PARALLEL_BATCHES = 4
_C.DATALOADER.NUM_PREFETCH_BATCHES = 2

_C.DATALOADER.LOAD_SEMANTIC_MASKS = False

# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.MODEL.BACKBONE = CN()

_C.MODEL.BACKBONE.NAME = "ResNet"
# Add StopGrad at a specified stage so the bottom layers are frozen
_C.MODEL.BACKBONE.FREEZE_AT = 2


# --------------------------------------------------------------------------- #
# Resnet options
# --------------------------------------------------------------------------- #
_C.MODEL.RESNETS = CN()

_C.MODEL.RESNETS.DEPTH = 101

# C4 for C4 backbone, res2..5 for FPN backbone
_C.MODEL.RESNETS.OUT_FEATURES = ["res4"]

# Number of groups to use; 1 ==> ResNet; > 1 ==> ResNeXt
_C.MODEL.RESNETS.NUM_GROUPS = 1

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.RESNETS.NORM = "FrozenBN"

_C.MODEL.RESNETS.ACTIVATION = "mish"

# Baseline width of each group.
# Scaling this parameters will scale the depth of all bottleneck layers.
_C.MODEL.RESNETS.WIDTH_PER_GROUP = 64

# Place the stride 2 conv on the 1x1 filter
# Use True only for the original MSRA ResNet; use False for res2 and Torch models
_C.MODEL.RESNETS.STRIDE_IN_1X1 = True

# Apply dilation in stage res5
_C.MODEL.RESNETS.RES5_DILATION = 1

# Output width of res2.
# Scaling this parameters will scale the width of all 1x1 convs in ResNet
_C.MODEL.RESNETS.RES2_OUT_CHANNELS = 256
_C.MODEL.RESNETS.STEM_OUT_CHANNELS = 64

# Apply Deformable Convolution in stages
# Specify if apply deform_conv on res2, res3, res4, res5
_C.MODEL.RESNETS.DEFORM_ON_PER_STAGE = [False, False, False, False]
# Use True to use modulated deform_conv
# (DeformableV2, https://arxiv.org/abs/1811.11168);
# Use False for DeformableV1.
_C.MODEL.RESNETS.DEFORM_MODULATED = False
# Number of groups in deformable conv.
_C.MODEL.RESNETS.DEFORM_NUM_GROUPS = 1


# --------------------------------------------------------------------------- #
# SpineNet options
# --------------------------------------------------------------------------- #
_C.MODEL.SPINENETS = CN()

_C.MODEL.SPINENETS.VARIANT = "49"

_C.MODEL.SPINENETS.OUT_FEATURES = ["sp3_2", "sp4_4", "sp5_4", "sp6_2", "sp7_2"]

# Options: FrozenBN, GN, "SyncBN", "BN"
_C.MODEL.SPINENETS.NORM = "FrozenBN"

_C.MODEL.SPINENETS.STEM_OUT_CHANNELS = 64
_C.MODEL.SPINENETS.L2_OUT_CHANNELS = 64

# one of `residual` or `bottleneck`
_C.MODEL.SPINENETS.INIT_BLOCK_TYPE = "bottleneck"
_C.MODEL.SPINENETS.NUM_INIT_BLOCKS = 2

_C.MODEL.SPINENETS.INIT_DROP_CONNECT_RATE = 0.

# one of `relu` or `swish`
_C.MODEL.SPINENETS.ACTIVATION = "swish"


# --------------------------------------------------------------------------- #
# Neck options
# --------------------------------------------------------------------------- #
_C.MODEL.NECK = CN()

_C.MODEL.NECK.NAME = ""

_C.MODEL.NECK.IN_FEATURES = []
_C.MODEL.NECK.OUT_CHANNELS = 256

# Options: "" (no norm), "GN"
_C.MODEL.NECK.NORM = ""

_C.MODEL.NECK.ACTIVATION = ""

# Types for fusing the FPN top-down and lateral features. Can be either "sum", "avg" or "concat"
_C.MODEL.NECK.FUSE_TYPE = "sum"

_C.MODEL.NECK.TOP_BLOCK_TYPE = "MAXPOOL"

# ---------------------------------------------------------------------------- #
# Proposal generator options
# ---------------------------------------------------------------------------- #
_C.MODEL.PROPOSAL_GENERATOR = CN()
# Current proposal generators include "RPN", "RRPN" and "PrecomputedProposals"
_C.MODEL.PROPOSAL_GENERATOR.NAME = "RPN"
# Proposal height and width both need to be greater than MIN_SIZE
# (a the scale used during training or inference)
_C.MODEL.PROPOSAL_GENERATOR.MIN_SIZE = 0


# --------------------------------------------------------------------------- #
# Anchor generator options
# --------------------------------------------------------------------------- #
_C.MODEL.ANCHOR_GENERATOR = CN()
# The generator can be any name in the ANCHOR_GENERATOR registry
_C.MODEL.ANCHOR_GENERATOR.NAME = "DefaultAnchorGenerator"
# anchor sizes given in absolute pixels w.r.t. the scaled network input.
# Format: list of lists of sizes. SIZES[i] specifies the list of sizes
# to use for IN_FEATURES[i]; len(SIZES) == len(IN_FEATURES) must be true,
# or len(SIZES) == 1 is true and size list SIZES[0] is used for all
# IN_FEATURES.
_C.MODEL.ANCHOR_GENERATOR.SIZES = [[32, 64, 128, 256, 512]]
# Anchor aspect ratios.
# Format is list of lists of sizes. ASPECT_RATIOS[i] specifies the list of
# aspect ratios to use for IN_FEATURES[i];
# len(ASPECT_RATIOS) == len(IN_FEATURES) must be true,
# or len(ASPECT_RATIOS) == 1 is true and aspect ratio list ASPECT_RATIOS[0]
# is used for all IN_FEATURES.
_C.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]
# Anchor angles.
# list[float], the angle in degrees, for each input feature map.
# ANGLES[i] specifies the list of angles for IN_FEATURES[i].
_C.MODEL.ANCHOR_GENERATOR.ANGLES = [[-90, 0, 90]]


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
_C.MODEL.RPN = CN()
_C.MODEL.RPN.HEAD_NAME = "StandardRPNHead"  # used by RPN_HEAD_REGISTRY

# Names of the input feature maps to be used by RPN
# e.g., ["p2", "p3", "p4", "p5", "p6"] for FPN
_C.MODEL.RPN.IN_FEATURES = ["res4"]
# Remove RPN anchors that go outside the image by BOUNDARY_THRESH pixels
# Set to -1 or a large value, e.g. 100000, to disable pruning anchors
_C.MODEL.RPN.BOUNDARY_THRESH = -1
# IOU overlap ratios [BG_IOU_THRESHOLD, FG_IOU_THRESHOLD]
# Minimum overlap required between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a positive example (IoU >= FG_IOU_THRESHOLD
# ==> positive RPN example: 1)
# Maximum overlap allowed between an anchor and ground-truth box for the
# (anchor, gt box) pair to be a negative examples (IoU < BG_IOU_THRESHOLD
# ==> negative RPN example: 0)
# Anchors with overlap in between (BG_IOU_THRESHOLD <= IoU < FG_IOU_THRESHOLD)
# are ignored (-1)
_C.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
_C.MODEL.RPN.IOU_LABELS = [0, -1, 1]
# Total number of RPN examples per image
_C.MODEL.RPN.BATCH_SIZE_PER_IMAGE = 256
# Target fraction of foreground (positive) examples per RPN minibatch
_C.MODEL.RPN.POSITIVE_FRACTION = 0.5
# Weights on (dx, dy, dw, dh) for normalizing RPN anchor regression targets
_C.MODEL.RPN.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)
# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
_C.MODEL.RPN.SMOOTH_L1_BETA = 0.0
_C.MODEL.RPN.LOSS_WEIGHT = 1.0
# Number of top scoring RPN proposals to keep before applying NMS
# When FPN is used, this is *per FPN level* (not total)
_C.MODEL.RPN.PRE_NMS_TOPK_TRAIN = 12000
_C.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
# Number of top scoring RPN proposals to keep after applying NMS
# When FPN is used, this limit is applied per level and then again to the union
# of proposals from all levels
# NOTE: When FPN is used, the meaning of this config is different from Detectron1.
# It means per-batch topk in Detectron1, but per-image topk here.
# See "modeling/rpn/rpn_outputs.py" for details.
_C.MODEL.RPN.POST_NMS_TOPK_TRAIN = 2000
_C.MODEL.RPN.POST_NMS_TOPK_TEST = 1000
# NMS threshold used on RPN proposals
_C.MODEL.RPN.NMS_THRESH = 0.7

# ---------------------------------------------------------------------------- #
# ROI HEADS options
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_HEADS = CN()
_C.MODEL.ROI_HEADS.NAME = "Res5ROIHeads"
# Number of foreground classes
_C.MODEL.ROI_HEADS.NUM_CLASSES = 80
# Names of the input feature maps to be used by ROI heads
# Currently all heads (box, mask, ...) use the same input feature map list
# e.g., ["p2", "p3", "p4", "p5"] is commonly used for FPN
_C.MODEL.ROI_HEADS.IN_FEATURES = ["res4"]
# IOU overlap ratios [IOU_THRESHOLD]
# Overlap threshold for an RoI to be considered background (if < IOU_THRESHOLD)
# Overlap threshold for an RoI to be considered foreground (if >= IOU_THRESHOLD)
_C.MODEL.ROI_HEADS.IOU_THRESHOLDS = [0.5]
_C.MODEL.ROI_HEADS.IOU_LABELS = [0, 1]
# RoI minibatch size *per image* (number of regions of interest [ROIs])
# Total number of RoIs per training minibatch =
#   ROI_HEADS.BATCH_SIZE_PER_IMAGE * SOLVER.IMS_PER_BATCH
# E.g., a common configuration is: 512 * 16 = 8192
_C.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
# Target fraction of RoI minibatch that is labeled foreground (i.e. class > 0)
_C.MODEL.ROI_HEADS.POSITIVE_FRACTION = 0.25
# If True, augment proposals with ground-truth boxes before sampling proposals to
# train ROI heads.
_C.MODEL.ROI_HEADS.PROPOSAL_APPEND_GT = True

# Only used on test mode

# Minimum score threshold (assuming scores in a [0, 1] range); a value chosen to
# balance obtaining high recall with not having too many low precision
# detections that will slow down inference post processing steps (like NMS)
# A default threshold of 0.0 increases AP by ~0.2-0.3 but significantly slows down
# inference.
_C.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
_C.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.5
_C.MODEL.ROI_HEADS.NMS_CLS_AGNOSTIC = False

# ---------------------------------------------------------------------------- #
# Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_HEAD = CN()
# C4 don't use head name option
# Options for non-C4 models: FastRCNNConvFCHead,
_C.MODEL.ROI_BOX_HEAD.NAME = ""
# Default weights on (dy, dx, dh, dw) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
_C.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS = (10.0, 10.0, 5.0, 5.0)
# The transition point from L1 to L2 loss. Set to 0.0 to make the loss simply L1.
_C.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA = 0.0
_C.MODEL.ROI_BOX_HEAD.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.ROI_BOX_HEAD.FOCAL_LOSS_GAMMA = 2.0

_C.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO = 0
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_BOX_HEAD.POOLER_TYPE = "ROIAlignV2"

_C.MODEL.ROI_BOX_HEAD.NUM_FC = 0
# Hidden layer dimension for FC layers in the RoI box head
_C.MODEL.ROI_BOX_HEAD.FC_DIM = 1024
_C.MODEL.ROI_BOX_HEAD.NUM_CONV = 0
# Channel dimension for Conv layers in the RoI box head
_C.MODEL.ROI_BOX_HEAD.CONV_DIM = 256
# Normalization method for the convolution layers.
# Options: "" (no norm), "GN", "SyncBN".
_C.MODEL.ROI_BOX_HEAD.NORM = ""
# Whether to use class agnostic for bbox regression
_C.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG = False

# ---------------------------------------------------------------------------- #
# Relation Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_RELATION_HEAD = CN()
_C.MODEL.ROI_BOX_RELATION_HEAD.NUM_GROUPS = 16
_C.MODEL.ROI_BOX_RELATION_HEAD.KEY_DIM = 64
_C.MODEL.ROI_BOX_RELATION_HEAD.GEOMETRY_EMBEDDING_DIM = 64

_C.MODEL.ROI_BOX_RELATION_HEAD.DUPLICATE_REMOVAL_IOU = 0.5
_C.MODEL.ROI_BOX_RELATION_HEAD.RANK_EMBEDDING_DIM = 128
_C.MODEL.ROI_BOX_RELATION_HEAD.NMS_NUM_GROUP = 16

# ---------------------------------------------------------------------------- #
# Cascaded Box Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_BOX_CASCADE_HEAD = CN()
# The number of cascade stages is implicitly defined by the length of the following two configs.
_C.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS = (
    (10.0, 10.0, 5.0, 5.0),
    (20.0, 20.0, 10.0, 10.0),
    (30.0, 30.0, 15.0, 15.0),
)
_C.MODEL.ROI_BOX_CASCADE_HEAD.IOUS = (0.5, 0.6, 0.7)

# ---------------------------------------------------------------------------- #
# Mask Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_MASK_HEAD = CN()
_C.MODEL.ROI_MASK_HEAD.NAME = "MaskRCNNConvUpsampleHead"
_C.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO = 0
_C.MODEL.ROI_MASK_HEAD.NUM_CONV = 0  # The number of convs in the mask head
_C.MODEL.ROI_MASK_HEAD.CONV_DIM = 256
# Normalization method for the convolution layers.
# Options: "" (no norm), "GN", "SyncBN".
_C.MODEL.ROI_MASK_HEAD.NORM = ""
# Whether to use class agnostic for mask prediction
_C.MODEL.ROI_MASK_HEAD.CLS_AGNOSTIC_MASK = False
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_MASK_HEAD.POOLER_TYPE = "ROIAlignV2"


# ---------------------------------------------------------------------------- #
# Keypoint Head
# ---------------------------------------------------------------------------- #
_C.MODEL.ROI_KEYPOINT_HEAD = CN()
_C.MODEL.ROI_KEYPOINT_HEAD.NAME = "KRCNNConvDeconvUpsampleHead"
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION = 14
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO = 2
_C.MODEL.ROI_KEYPOINT_HEAD.CONV_DIMS = tuple(512 for _ in range(8))
_C.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = 17  # 17 is the number of keypoints in COCO.

# Images with too few (or no) keypoints are excluded from training.
_C.MODEL.ROI_KEYPOINT_HEAD.MIN_KEYPOINTS_PER_IMAGE = 1
# Normalize by the total number of visible keypoints in the minibatch if True.
# Otherwise, normalize by the total number of keypoints that could ever exist
# in the minibatch.
# The keypoint softmax loss is only calculated on visible keypoints.
# Since the number of visible keypoints can vary significantly between
# minibatches, this has the effect of up-weighting the importance of
# minibatches with few visible keypoints. (Imagine the extreme case of
# only one visible keypoint versus N: in the case of N, each one
# contributes 1/N to the gradient compared to the single keypoint
# determining the gradient direction). Instead, we can normalize the
# loss by the total number of keypoints, if it were the case that all
# keypoints were visible in a full minibatch. (Returning to the example,
# this means that the one visible keypoint contributes as much as each
# of the N keypoints.)
_C.MODEL.ROI_KEYPOINT_HEAD.NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS = True
# Multi-task loss weight to use for keypoints
# Recommended values:
#   - use 1.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is True
#   - use 4.0 if NORMALIZE_LOSS_BY_VISIBLE_KEYPOINTS is False
_C.MODEL.ROI_KEYPOINT_HEAD.LOSS_WEIGHT = 1.0
# Type of pooling operation applied to the incoming feature map for each RoI
_C.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE = "ROIAlignV2"

# ---------------------------------------------------------------------------- #
# Semantic Segmentation Head
# ---------------------------------------------------------------------------- #
_C.MODEL.SEM_SEG_HEAD = CN()
_C.MODEL.SEM_SEG_HEAD.NAME = "SemSegFPNHead"
_C.MODEL.SEM_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
# Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
# the correposnding pixel.
_C.MODEL.SEM_SEG_HEAD.IGNORE_VALUE = -1
# Number of classes in the semantic segmentation head
_C.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 54
# Number of channels in the 3x3 convs inside semantic-FPN heads.
_C.MODEL.SEM_SEG_HEAD.CONVS_DIM = 128
# Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
_C.MODEL.SEM_SEG_HEAD.COMMON_STRIDE = 4
# Normalization method for the convolution layers. Options: "" (no norm), "GN".
_C.MODEL.SEM_SEG_HEAD.NORM = "GN"
_C.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT = 1.0

_C.MODEL.PANOPTIC_FPN = CN()
# Scaling of all losses from instance detection / segmentation head.
_C.MODEL.PANOPTIC_FPN.INSTANCE_LOSS_WEIGHT = 1.0

# options when combining instance & semantic segmentation outputs
_C.MODEL.PANOPTIC_FPN.COMBINE = CN({"ENABLED": True})
_C.MODEL.PANOPTIC_FPN.COMBINE.OVERLAP_THRESH = 0.5
_C.MODEL.PANOPTIC_FPN.COMBINE.STUFF_AREA_LIMIT = 4096
_C.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5


# ---------------------------------------------------------------------------- #
# Single Stage Detector Head
# ---------------------------------------------------------------------------- #
_C.MODEL.SINGLE_STAGE_HEAD = CN()

_C.MODEL.SINGLE_STAGE_HEAD.NAME = "RetinaNetHead"

# This is the number of foreground classes.
_C.MODEL.SINGLE_STAGE_HEAD.NUM_CLASSES = 80

_C.MODEL.SINGLE_STAGE_HEAD.IN_FEATURES = ["p3", "p4", "p5", "p6", "p7"]

_C.MODEL.SINGLE_STAGE_HEAD.IOU_THRESHOLDS = [0.4, 0.5]
_C.MODEL.SINGLE_STAGE_HEAD.IOU_LABELS = [0, -1, 1]

# ---------------------------------------------------------------------------- #
# RetinaNet
# ---------------------------------------------------------------------------- #
_C.MODEL.RETINANET = CN()

# Convolutions to use in the conv tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.RETINANET.NUM_CONVS = 4

# IoU overlap ratio [bg, fg] for labeling anchors.
# Anchors with < bg are labeled negative (0)
# Anchors  with >= bg and < fg are ignored (-1)
# Anchors with >= fg are labeled positive (1)

# Prior prob for rare case (i.e. foreground) at the beginning of training.
# This is used to set the bias for the logits layer of the classifier subnet.
# This improves training stability in the case of heavy class imbalance.
_C.MODEL.RETINANET.PRIOR_PROB = 0.01

# Inference cls score threshold, only anchors with score > INFERENCE_TH are
# considered for inference (to improve speed)
_C.MODEL.RETINANET.SCORE_THRESH_TEST = 0.05
_C.MODEL.RETINANET.TOPK_CANDIDATES_TEST = 1000
_C.MODEL.RETINANET.NMS_THRESH_TEST = 0.5
_C.MODEL.RETINANET.NMS_CLS_AGNOSTIC = False

# Weights on (dx, dy, dw, dh) for normalizing anchor regression targets
_C.MODEL.RETINANET.BBOX_REG_WEIGHTS = (1.0, 1.0, 1.0, 1.0)

# Loss parameters
_C.MODEL.RETINANET.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.RETINANET.FOCAL_LOSS_ALPHA = 0.25
_C.MODEL.RETINANET.SMOOTH_L1_LOSS_BETA = 0.1

# ---------------------------------------------------------------------------- #
# SOLO
# ---------------------------------------------------------------------------- #
_C.MODEL.SOLO = CN()

# Convolutions to use in the conv tower
# NOTE: this doesn't include the last conv for logits
_C.MODEL.SOLO.MASK_KERNEL_NUM_CONVS = 4
_C.MODEL.SOLO.USE_DEFORM_CONV = False
_C.MODEL.SOLO.DEFORM_MODULATED = False
_C.MODEL.SOLO.MASK_KERNEL_NORM = "GN"
_C.MODEL.SOLO.MASK_KERNEL_SIZE = 1
_C.MODEL.SOLO.MASK_KERNEL_CONVS_DIM = 512

# Number of channels in the 3x3 convs inside mask feature branch.
_C.MODEL.SOLO.MASK_FEATURE_IN_FEATURES = ["p2", "p3", "p4", "p5"]
_C.MODEL.SOLO.MASK_FEATURE_CONVS_DIM = 128
_C.MODEL.SOLO.MASK_FEATURE_OUT_DIMS = 256
_C.MODEL.SOLO.MASK_FEATURE_COMMON_STRIDE = 4
_C.MODEL.SOLO.MASK_FEATURE_NORM = "GN"

_C.MODEL.SOLO.SCALE_RANGES = [[1, 96], [48, 192], [96, 384], [192, 768], [384, 2048]]
_C.MODEL.SOLO.NUM_GRIDS = [40, 36, 24, 16, 12]

# Prior prob for rare case (i.e. foreground) at the beginning of training.
# This is used to set the bias for the logits layer of the classifier subnet.
# This improves training stability in the case of heavy class imbalance.
_C.MODEL.SOLO.PRIOR_PROB = 0.01

# Center sampling
_C.MODEL.SOLO.SIGMA = 0.2

# Loss parameters

# Focal loss
_C.MODEL.SOLO.FOCAL_LOSS_GAMMA = 2.0
_C.MODEL.SOLO.FOCAL_LOSS_ALPHA = 0.25
# Dice loss
_C.MODEL.SOLO.INS_LOSS_WEIGHT = 3.0

# Inference cls score threshold, only anchors with score > INFERENCE_TH are
# considered for inference (to improve speed)
_C.MODEL.SOLO.SCORE_THRESH_TEST = 0.1
_C.MODEL.SOLO.UPDATE_SCORE_THRESH_TEST = 0.05
_C.MODEL.SOLO.MASK_THRESH_TEST = 0.5
_C.MODEL.SOLO.TOPK_CANDIDATES_TEST = 500
_C.MODEL.SOLO.NMS_KERNEL = "gaussian"  # gaussian or linear
_C.MODEL.SOLO.NMS_SIGMA = 2.0
_C.MODEL.SOLO.NMS_CLS_AGNOSTIC = False

_C.MODEL.YOLOV4 = CN()

_C.MODEL.YOLOV4.CONV_DIMS = 256
_C.MODEL.YOLOV4.NORM = "BN"
_C.MODEL.YOLOV4.ACTIVATION = "leaky_relu"

_C.MODEL.YOLOV4.SCALE_YX = [1.2, 1.1, 1.05]
_C.MODEL.YOLOV4.CLS_NORMALIZER = 1.0
_C.MODEL.YOLOV4.IOU_NORMALIZER = 0.07

_C.MODEL.YOLOV4.SCORE_THRESH_TEST = 0.05
_C.MODEL.YOLOV4.NMS_THRESH_TEST = 0.5

# ---------------------------------------------------------------------------- #
# Solver
# ---------------------------------------------------------------------------- #
_C.SOLVER = CN()

# See detectron2/solver/build.py for LR scheduler options
_C.SOLVER.LR_SCHEDULER_NAME = "WarmupMultiStepLR"

# Number of images per batch across all machines.
# If we have 8 GPUs and IMS_PER_BATCH = 16,
# each GPU will see 2 images per batch.
_C.SOLVER.NUM_GPUS = 8
_C.SOLVER.IMS_PER_GPU = 2
_C.SOLVER.IMS_PER_BATCH = 16

_C.SOLVER.AUTO_SCALE_LR_SCHEDULE = True
_C.SOLVER.IMS_PER_BATCH_BASE = 16

_C.SOLVER.MAX_ITER = 40000

# Steps to save the model.
_C.SOLVER.SHORT_TERM_NUM_STEPS = 10000
_C.SOLVER.SHORT_TERM_SAVE_STEPS = 2000
_C.SOLVER.LONG_TERM_SAVE_STEPS = 10000

_C.SOLVER.BASE_LR = 0.001

_C.SOLVER.MOMENTUM = 0.9

_C.SOLVER.WEIGHT_DECAY = 0.0001
# The weight decay that's applied to parameters of normalization layers
# (typically the affine transformation)
_C.SOLVER.WEIGHT_DECAY_NORM = 0.0

_C.SOLVER.GAMMA = 0.1
_C.SOLVER.STEPS = (30000,)

_C.SOLVER.WARMUP_FACTOR = 1.0 / 1000
_C.SOLVER.WARMUP_ITERS = 1000
_C.SOLVER.WARMUP_METHOD = "linear"

_C.SOLVER.CHECKPOINT_PERIOD = 5000

# Detectron v1 (and previous detection code) used a 2x higher LR and 0 WD for
# biases. This is not useful (at least for recent models). You should avoid
# changing these and they exist only to reproduce Detectron v1 training if
# desired.
_C.SOLVER.BIAS_LR_FACTOR = 1.0
_C.SOLVER.WEIGHT_DECAY_BIAS = _C.SOLVER.WEIGHT_DECAY

_C.SOLVER.CLIP_GRADIENTS_BY_NORM = 10.0

# ---------------------------------------------------------------------------- #
# Specific test options
# ---------------------------------------------------------------------------- #
_C.TEST = CN()
# For end-to-end tests to verify the expected accuracy.
# Each item is [task, metric, value, tolerance]
# e.g.: [['bbox', 'AP', 38.5, 0.2]]
_C.TEST.EXPECTED_RESULTS = []
# The period (in terms of steps) to evaluate the model during training.
# Set to 0 to disable.
_C.TEST.EVAL_PERIOD = 0
# The sigmas used to calculate keypoint OKS.
# When empty it will use the defaults in COCO.
# Otherwise it should have the same length as ROI_KEYPOINT_HEAD.NUM_KEYPOINTS.
_C.TEST.KEYPOINT_OKS_SIGMAS = []
# Maximum number of detections to return per image during inference (100 is
# based on the limit established for the COCO dataset).
_C.TEST.DETECTIONS_PER_IMAGE = 100

_C.TEST.AUG = CN({"ENABLED": False})
_C.TEST.AUG.MIN_SIZES = (400, 500, 600, 700, 800, 900, 1000, 1100, 1200)
_C.TEST.AUG.MAX_SIZE = 4000
_C.TEST.AUG.FLIP = True

_C.TEST.PRECISE_BN = CN({"ENABLED": False})
_C.TEST.PRECISE_BN.NUM_ITER = 200

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Directory where output files are written
_C.OUTPUT_DIR = "./output"
# Set seed to negative to fully randomize everything.
# Set seed to positive to use a fixed seed. Note that a fixed seed does not
# guarantee fully deterministic behavior.
_C.SEED = -1
# Benchmark different cudnn algorithms. It has large overhead for about 10k
# iterations. It usually hurts total time, but can benefit for certain models.
_C.CUDNN_BENCHMARK = False

# global config is for quick hack purposes.
# You can set them in command line or config files,
# and access it with:
#
# from detectron2.config import global_cfg
# print(global_cfg.HACK)
#
# Do not commit any configs into it.
_C.GLOBAL = CN()
_C.GLOBAL.HACK = 1.0
