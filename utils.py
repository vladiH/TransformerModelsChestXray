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
                  'lr_scheduler': lr_scheduler.state_dict(),
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
