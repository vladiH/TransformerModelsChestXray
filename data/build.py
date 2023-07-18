# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import numpy as np
import torch.distributed as dist
import random
from torchvision import datasets, transforms
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from PIL import ImageEnhance

from torch.utils.data import Subset
from .custom_image_folder import MyImageFolder
from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler

def build_normal_loader(config, percent):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    (dataset_val, dataset_test), _ = build_dataset(is_train=False, config=config)



    # Create a random subset of 100 elements
    total_elements = len(dataset_train)
    random_indices = random.sample(range(total_elements), int(total_elements*percent))
    subset_dataset_train = Subset(dataset_train, random_indices)
    data_loader_train = torch.utils.data.DataLoader(
        subset_dataset_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY, #https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        drop_last=True,
    )

    # Create a random subset of 100 elements
    total_elements = len(dataset_val)
    random_indices = random.sample(range(total_elements), int(total_elements*percent))
    subset_dataset_val = Subset(dataset_val, random_indices)
    data_loader_val = torch.utils.data.DataLoader(
        subset_dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY, #https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        drop_last=False
    )

    # Create a random subset of 100 elements
    total_elements = len(dataset_test)
    random_indices = random.sample(range(total_elements), int(total_elements*percent))
    subset_dataset_test = Subset(dataset_test, random_indices)
    data_loader_test = torch.utils.data.DataLoader(
        subset_dataset_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,#https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn


def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    (dataset_val, dataset_test), _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val and test dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
        indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
        sampler_train = SubsetRandomSampler(indices)
    else:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY, #https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        drop_last=True,
    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY, #https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        drop_last=False
    )

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        batch_size=config.DATA.BATCH_SIZE,
        shuffle=False,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,#https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/
        drop_last=False
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    elif config.DATA.DATASET == 'nih':
        trainset = MyImageFolder(root=config.NIH.trainset, csv_path=config.NIH.train_csv_path, transform=transform)
        validset = MyImageFolder(root=config.NIH.validset, csv_path=config.NIH.valid_csv_path, transform=transform)
        testset = MyImageFolder(root=config.NIH.testset, csv_path=config.NIH.test_csv_path, transform=transform)
        dataset = trainset if is_train else (validset, testset)
        nb_classes = 14
    else:
        raise NotImplementedError("We only support ImageNet and NIH Now.")

    return dataset, nb_classes

# Custom transformations
def apply_custom_transforms(image):
    # Sharpening
    enhancer = ImageEnhance.Sharpness(image)
    sharpened_image = enhancer.enhance(random.uniform(0.5, 1.5))
    return sharpened_image

def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        if config.AUG.AUTO_AUGMENT!='none':
            # this should always dispatch to transforms_imagenet_train
            transform = create_transform(
                input_size=config.DATA.IMG_SIZE,
                is_training=True,
                color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
                auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
                re_prob=config.AUG.REPROB,
                re_mode=config.AUG.REMODE,
                re_count=config.AUG.RECOUNT,
                interpolation=config.DATA.INTERPOLATION,
            )
        else:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(size=config.DATA.IMG_SIZE, scale=(0.8, 1.0), ratio=(0.75, 1.3333), interpolation=get_interpolation_mode(config.DATA.INTERPOLATION)),
                transforms.RandomApply([transforms.RandomRotation(degrees=(10, 175), interpolation=get_interpolation_mode(config.DATA.INTERPOLATION)),
                                        transforms.RandomHorizontalFlip(p=0.5), 
                                        # transforms.RandomVerticalFlip(p=0.5), 
                                        transforms.RandomAffine(degrees=(15, 15), translate=(0.1, 0.3), scale=(0.8, 1.0), interpolation=get_interpolation_mode(config.DATA.INTERPOLATION)),
                                        transforms.RandomPerspective(distortion_scale=0.35, p=0.5, interpolation=get_interpolation_mode(config.DATA.INTERPOLATION)),
                                        transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5)),
                                        transforms.RandomAutocontrast(p=0.5),
                                        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
                                        ], p=0.5),
                transforms.ToTensor(),
                transforms.RandomApply([apply_noise], p=0.25),
                transforms.RandomErasing(p=config.AUG.REPROB, scale=(0.02,0.02), inplace=True),
                transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
            ])
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=get_interpolation_mode(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=get_interpolation_mode(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)

def get_interpolation_mode(text):
    if text.lower() == "random":
        return InterpolationMode.BICUBIC
    elif text.lower() == "bilinear":
        return InterpolationMode.BILINEAR
    elif text.lower() == "bicubic":
        return InterpolationMode.BICUBIC
    elif text.lower() == "nearest":
        return InterpolationMode.NEAREST
    else:
        raise ValueError("Invalid interpolation mode: " + text)

# Custom transformations
# def apply_rotation(image):
#     angle = random.uniform(-10, 10)
#     return transforms.transforms.F.rotate(image, angle, fill=0)

# def apply_flip(image):
#     if random.random() < 0.5:
#         return transforms.transforms.F.hflip(image)
#     return image

# def apply_shift(image):
#     shift_x = random.uniform(-10, 10)
#     shift_y = random.uniform(-10, 10)
#     return transforms.transforms.F.affine(image, angle=0, translate=(shift_x, shift_y), scale=1, shear=0, fill=0)

# def apply_zoom(image):
#     zoom_factor = random.uniform(0.8, 1.2)
#     return transforms.transforms.F.resize(image, size=int(image.width * zoom_factor), interpolation=InterpolationMode.BICUBIC)

# def apply_shear(image):
#     shear_x = random.uniform(-0.2, 0.2)
#     shear_y = random.uniform(-0.2, 0.2)
#     return transforms.transforms.F.affine(image, angle=0, translate=(0, 0), scale=1, shear=(shear_x, shear_y), fill=0)

def apply_noise(image):
    noise = torch.randn_like(image) * 0.01
    return image + noise

# def apply_gaussian_filter(image):
#     return transforms.transforms.F.gaussian_blur(image, kernel_size=3)

# def apply_contrast(image):
#     contrast_factor = random.uniform(0.8, 1.2)
#     return transforms.transforms.F.adjust_contrast(image, contrast_factor)

# def apply_brightness(image):
#     brightness_factor = random.uniform(0.8, 1.2)
#     return transforms.transforms.F.adjust_brightness(image, brightness_factor)

# def apply_sharpening(image):
#     sharpened_image = transforms.transforms.F.adjust_sharpness(image, 2.0)
#     return sharpened_image