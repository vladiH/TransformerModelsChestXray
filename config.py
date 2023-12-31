# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 32
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
# _C.DATA.DATASET = 'imagenet'
_C.DATA.DATASET = 'nih'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic, nearest)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 16

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type, it would be swin or maxvit
_C.MODEL.TYPE = 'maxvit'
# Model name
_C.MODEL.NAME = 'maxvit_tiny_tf_224.in1k'
# Pretrained weight from checkpoint, could be imagenet22k pretrained weight
# could be overwritten by command line argument
_C.MODEL.PRETRAINED = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
# _C.MODEL.NUM_CLASSES = 1000
_C.MODEL.NUM_CLASSES = 14
# Dropout rate, available for swin
_C.MODEL.DROP_RATE = 0.0
# Drop path rate, available for swin
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# Vit transformer parameters
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PATCH_SIZE = 16
_C.MODEL.VIT.IN_CHANS = 3
_C.MODEL.VIT.EMBED_DIM = 768
_C.MODEL.VIT.DEPTHS = 12
_C.MODEL.VIT.NUM_HEADS = 12

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

#Maxvit Transformer parameters
_C.MODEL.MAXVIT = CN()
_C.MODEL.MAXVIT.IN_CHANS = 3
# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
#transformer
# _C.TRAIN.BASE_LR = 3e-3
# _C.TRAIN.WARMUP_LR = 5e-6
# _C.TRAIN.MIN_LR = 5e-4
#swin
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = False
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

#Early stopping,
_C.TRAIN.EARLYSTOPPING = CN()

#auc, loss or None
_C.TRAIN.EARLYSTOPPING.MONITOR = 'auc'
_C.TRAIN.EARLYSTOPPING.PATIENCE = 5

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.0#0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m5-p0.5-mstd0.5-inc1' #'rand-m6-mstd0.5-inc1' #'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
# _C.AUG.MIXUP = 0.8
_C.AUG.MIXUP = 0
# Cutmix alpha, cutmix enabled if > 0
# _C.AUG.CUTMIX = 1.0
_C.AUG.CUTMIX = 0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if False, no amp is used
# overwritten by command line argument
_C.AMP_OPT_LEVEL = False
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
# Frequency to logging info
_C.PRINT_FREQ = 100
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0
# max checkpoints to preserve 
_C.MAX_CHECKPOINTS = 3

#nih
_C.NIH = CN()
_C.NIH.trainset = ''
_C.NIH.validset = ''
_C.NIH.testset = ''
# _C.NIH.class_num = -1
_C.NIH.train_csv_path = ''
_C.NIH.valid_csv_path = ''
_C.NIH.test_csv_path = ''
_C.NIH.num_mlp_heads = 3



def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args, ommit=False):
    _update_config_from_file(config, args.cfg)
    if not ommit:
        config.defrost()
        if args.opts:
            config.merge_from_list(args.opts)

        # merge from specific arguments
        if args.pretrained:
            config.MODEL.PRETRAINED = args.pretrained
        if args.batch_size:
            config.DATA.BATCH_SIZE = args.batch_size
        if args.data_path:
            config.DATA.DATA_PATH = args.data_path
        if args.zip:
            config.DATA.ZIP_MODE = True
        if args.cache_mode:
            config.DATA.CACHE_MODE = args.cache_mode
        if args.resume:
            config.MODEL.RESUME = args.resume
        if args.accumulation_steps:
            config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
        if args.use_checkpoint:
            config.TRAIN.USE_CHECKPOINT = True
        if args.amp_opt_level:
            config.AMP_OPT_LEVEL = True
        if args.output:
            config.OUTPUT = args.output
        if args.tag:
            config.TAG = args.tag
        if args.eval:
            config.EVAL_MODE = True
        if args.throughput:
            config.THROUGHPUT_MODE = True

        # set local rank for distributed training
        config.LOCAL_RANK = int(os.environ["LOCAL_RANK"]) if "LOCAL_RANK" in os.environ else None

        # output folder
        config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

        # nih
        config.NIH.trainset = args.trainset
        config.NIH.validset = args.validset
        config.NIH.testset = args.testset
        # config.NIH.class_num = args.class_num
        config.NIH.train_csv_path = args.train_csv_path
        config.NIH.valid_csv_path = args.valid_csv_path
        config.NIH.test_csv_path = args.test_csv_path
        config.NIH.num_mlp_heads = args.num_mlp_heads

        config.freeze()


def get_config(args, ommit=False):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args, ommit)

    return config
