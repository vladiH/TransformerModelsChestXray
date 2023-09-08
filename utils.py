# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import torch
import torch.distributed as dist
import numpy as np

def load_checkpoint(config, model, optimizer, lr_scheduler, logger, scaler):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    sd = model.state_dict()
    for key, value in checkpoint['model'].items():
        if key != 'head.weight' and key != 'head.bias':     # TODO should comment this line when continuing from checkpoint (not pretrain)
            sd[key] = value
    model.load_state_dict(sd)
    # checkpoint['model']['head.weight'] = torch.zeros(2, model.num_features)
    # checkpoint['model']['head.bias'] = torch.zeros(2)
    # msg = model.load_state_dict(checkpoint['model'], strict=False)
    # logger.info(msg)
    logger.info("Pretrain model loaded successfully!")
    max_auc = 0.0
    min_loss = np.inf
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint['lr_scheduler'] is not None:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL and checkpoint['config'].AMP_OPT_LEVEL:
            scaler.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_auc' in checkpoint:
            max_auc = checkpoint['max_auc']
        if 'min_loss' in checkpoint:
            min_loss = checkpoint['min_loss']

    del checkpoint
    torch.cuda.empty_cache()
    return max_auc, min_loss


def save_checkpoint(config, epoch, model, max_auc, min_loss, optimizer, lr_scheduler, logger, scaler):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
                  'max_auc': max_auc,
                  'min_loss': min_loss,
                  'epoch': epoch,
                  'config': config}
    if config.AMP_OPT_LEVEL:
        save_state['amp'] = scaler.state_dict()

    save_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{epoch}.pth')
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")

    # Remove the oldest checkpoint if the number of saved checkpoints exceeds the limit
    checkpoint_files = os.listdir(config.OUTPUT)
    checkpoint_files = [f for f in checkpoint_files if f.endswith('.pth')]
    checkpoint_files = sorted(checkpoint_files, key=lambda x: int(x.split('_')[2].split('.')[0]))
    if len(checkpoint_files) > config.MAX_CHECKPOINTS:
        oldest_checkpoint_file = checkpoint_files[0]
        oldest_checkpoint_path = os.path.join(config.OUTPUT, oldest_checkpoint_file)
        os.remove(oldest_checkpoint_path)
        logger.info(f"Removed oldest checkpoint: {oldest_checkpoint_path}")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

def load_pretrained(config, model, logger):
    logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
    checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
    state_dict = checkpoint['model']
    pos_embed_index_keys = [k for k in state_dict.keys() if "pos_embed" in k]
    for k in pos_embed_index_keys:
        del state_dict[k]

    msg = model.load_state_dict(state_dict, strict=False)
    logger.warning(msg)

    logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

    del checkpoint
    torch.cuda.empty_cache()

#TODO: Uncomment to load pretrain SWIN MODEL
# def load_pretrained(config, model, logger):
#     logger.info(f"==============> Loading weight {config.MODEL.PRETRAINED} for fine-tuning......")
#     checkpoint = torch.load(config.MODEL.PRETRAINED, map_location='cpu')
#     state_dict = checkpoint['model']
#     # print(state_dict.keys())
#     # print(model.state_dict().keys())
#     # delete relative_position_index since we always re-init it
#     relative_position_index_keys = [k for k in state_dict.keys() if "relative_position_index" in k]
#     for k in relative_position_index_keys:
#         del state_dict[k]

#     # delete relative_coords_table since we always re-init it
#     relative_position_index_keys = [k for k in state_dict.keys() if "relative_coords_table" in k]
#     for k in relative_position_index_keys:
#         del state_dict[k]

#     # delete attn_mask since we always re-init it
#     attn_mask_keys = [k for k in state_dict.keys() if "attn_mask" in k]
#     for k in attn_mask_keys:
#         del state_dict[k]

#     # bicubic interpolate relative_position_bias_table if not match
#     relative_position_bias_table_keys = [k for k in state_dict.keys() if "relative_position_bias_table" in k]
#     for k in relative_position_bias_table_keys:
#         relative_position_bias_table_pretrained = state_dict[k]
#         relative_position_bias_table_current = model.state_dict()[k]
#         L1, nH1 = relative_position_bias_table_pretrained.size()
#         L2, nH2 = relative_position_bias_table_current.size()
#         if nH1 != nH2:
#             logger.warning(f"Error in loading {k}, passing......")
#         else:
#             if L1 != L2:
#                 # bicubic interpolate relative_position_bias_table if not match
#                 S1 = int(L1 ** 0.5)
#                 S2 = int(L2 ** 0.5)
#                 relative_position_bias_table_pretrained_resized = torch.nn.functional.interpolate(
#                     relative_position_bias_table_pretrained.permute(1, 0).view(1, nH1, S1, S1), size=(S2, S2),
#                     mode='bicubic')
#                 state_dict[k] = relative_position_bias_table_pretrained_resized.view(nH2, L2).permute(1, 0)

#     # bicubic interpolate absolute_pos_embed if not match
#     absolute_pos_embed_keys = [k for k in state_dict.keys() if "absolute_pos_embed" in k]
#     for k in absolute_pos_embed_keys:
#         # dpe
#         absolute_pos_embed_pretrained = state_dict[k]
#         absolute_pos_embed_current = model.state_dict()[k]
#         _, L1, C1 = absolute_pos_embed_pretrained.size()
#         _, L2, C2 = absolute_pos_embed_current.size()
#         if C1 != C1:
#             logger.warning(f"Error in loading {k}, passing......")
#         else:
#             if L1 != L2:
#                 S1 = int(L1 ** 0.5)
#                 S2 = int(L2 ** 0.5)
#                 absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.reshape(-1, S1, S1, C1)
#                 absolute_pos_embed_pretrained = absolute_pos_embed_pretrained.permute(0, 3, 1, 2)
#                 absolute_pos_embed_pretrained_resized = torch.nn.functional.interpolate(
#                     absolute_pos_embed_pretrained, size=(S2, S2), mode='bicubic')
#                 absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.permute(0, 2, 3, 1)
#                 absolute_pos_embed_pretrained_resized = absolute_pos_embed_pretrained_resized.flatten(1, 2)
#                 state_dict[k] = absolute_pos_embed_pretrained_resized

#     # check classifier, if not match, then re-init classifier to zero
#     # head_bias_pretrained = state_dict['head.bias']
#     # Nc1 = head_bias_pretrained.shape[0]
#     # Nc2 = model.head.bias.shape[0]
#     # if (Nc1 != Nc2):
#     #     if Nc1 == 21841 and Nc2 == 1000:
#     #         logger.info("loading ImageNet-22K weight to ImageNet-1K ......")
#     #         map22kto1k_path = f'data/map22kto1k.txt'
#     #         with open(map22kto1k_path) as f:
#     #             map22kto1k = f.readlines()
#     #         map22kto1k = [int(id22k.strip()) for id22k in map22kto1k]
#     #         state_dict['head.weight'] = state_dict['head.weight'][map22kto1k, :]
#     #         state_dict['head.bias'] = state_dict['head.bias'][map22kto1k]
#     #     else:
#     #         torch.nn.init.constant_(model.head.bias, 0.)
#     #         torch.nn.init.constant_(model.head.weight, 0.)
#     #         del state_dict['head.weight']
#     #         del state_dict['head.bias']
#     #         logger.warning(f"Error in loading classifier head, re-init classifier head to 0")

#     msg = model.load_state_dict(state_dict, strict=False)
#     logger.warning(msg)

#     logger.info(f"=> loaded successfully '{config.MODEL.PRETRAINED}'")

#     del checkpoint
#     torch.cuda.empty_cache()