# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import time
import argparse
import datetime
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.functional as F

import thop

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor, load_pretrained
from sklearn.metrics import roc_auc_score

from tqdm import tqdm
#https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
#https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
from torch.cuda.amp import autocast, GradScaler 

def parse_option():
    parser = argparse.ArgumentParser('MaxVit and Swin Transformer training and evaluation script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs. ",
        default=None,
        nargs='+',
    )

    # easy config modification
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument('--data-path', type=str, help='path to dataset')
    parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                             'full: cache all data, '
                             'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--pretrained',
                        help='pretrained weight from checkpoint, could be imagenet22k pretrained weight')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', action='store_true',
                        help='mixed precision opt level, if False, no amp is used')
    parser.add_argument('--output', default='output', type=str, metavar='PATH',
                        help='root of output folder, the full path is <output>/<model_name>/<tag> (default: output)')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true', help='Test throughput only')

    # distributed training
    # parser.add_argument("--local_rank", type=int, required=True, help='local rank for DistributedDataParallel')

    #nih
    parser.add_argument("--trainset", type=str, required=True, help='path to train dataset')
    parser.add_argument("--validset", type=str, required=True, help='path to validation dataset')
    parser.add_argument("--testset", type=str, required=True, help='path to test dataset')
    # parser.add_argument("--class_num", required=True, type=int,
    #                     help="Class number for binary classification, 0-13 for nih")
    parser.add_argument("--train_csv_path", type=str, required=True, help='path to train csv file')
    parser.add_argument("--valid_csv_path", type=str, required=True, help='path to validation csv file')
    parser.add_argument("--test_csv_path", type=str, required=True, help='path to test csv file')
    parser.add_argument("--num_mlp_heads", type=int, default=3, choices=[0, 1, 2, 3],
                        help='number of mlp layers at end of network')

    args, unparsed = parser.parse_known_args()

    config = get_config(args)

    return args, config


def main(config):
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = build_loader(config)

    logger.info(f"Creating model:{config.MODEL.TYPE}/{config.MODEL.NAME}")
    model = build_model(config)
    model.cuda()
    logger.info(str(model))
    optimizer = build_optimizer(config, model)
    scaler = GradScaler(enabled=config.AMP_OPT_LEVEL)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[config.LOCAL_RANK], broadcast_buffers=False)
    model_without_ddp = model.module

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"number of params: {n_parameters}")
    # if hasattr(model_without_ddp, 'flops'):
    #     flops = model_without_ddp.flops()
    #     logger.info(f"number of GFLOPs: {flops / 1e9}")

    flops(model_without_ddp, logger=logger, config=config)

    lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))

    if config.AUG.MIXUP > 0.:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif config.MODEL.LABEL_SMOOTHING > 0.:
        criterion = LabelSmoothingCrossEntropy(smoothing=config.MODEL.LABEL_SMOOTHING)
    else:
        criterion = torch.nn.CrossEntropyLoss()

    max_auc = 0.0
    min_loss = np.inf
    if config.TRAIN.AUTO_RESUME:
        resume_file = auto_resume_helper(config.OUTPUT)
        if resume_file:
            if config.MODEL.RESUME:
                logger.warning(f"auto-resume changing resume file from {config.MODEL.RESUME} to {resume_file}")
            config.defrost()
            config.MODEL.RESUME = resume_file
            config.freeze()
            logger.info(f'auto resuming from {resume_file}')
        else:
            logger.info(f'no checkpoint found in {config.OUTPUT}, ignoring auto resume')

    if config.MODEL.RESUME:
        max_auc, min_loss = load_checkpoint(config, model_without_ddp, optimizer, lr_scheduler, logger, scaler)
        acc1, acc5, auc, loss = validate(config, data_loader_val, model, is_validation=True)
        logger.info(f"Mean Accuracy of the network on the {len(dataset_val)} validation images: {acc1:.2f}%")
        logger.info(f"Mean AUC of the network on the {len(dataset_val)} validation images: {auc:.2f}%")
        logger.info(f"Mean Loss of the network on the {len(dataset_val)} validation images: {loss:.5f}")
        acc1, acc5, auc, loss = validate(config, data_loader_test, model, is_validation=False)
        logger.info(f"Mean Accuracy of the network on the {len(dataset_test)} test images: {acc1:.2f}%")
        logger.info(f"Mean AUC of the network on the {len(dataset_test)} test images: {auc:.2f}%")
        logger.info(f"Mean Loss of the network on the {len(dataset_test)} test images: {loss:.5f}")
        if config.EVAL_MODE:
            return

    if config.MODEL.PRETRAINED and (not config.MODEL.RESUME):
        load_pretrained(config, model_without_ddp, logger)
        # acc1, acc5, auc, loss = validate(config, data_loader_val, model, is_validation=True)
        # logger.info(f"AUC of the network on the {len(dataset_val)} val images: {auc:.1f}%")
        
    if config.THROUGHPUT_MODE:
        # throughput(data_loader_val, model, logger)
        throughput(data_loader_test, model, logger)
        return

    logger.info("Start training")
    if config.TRAIN.EARLYSTOPPING.MONITOR is not None:
        logger.info(f"Early Stopping is monitoring: {config.TRAIN.EARLYSTOPPING.MONITOR}")
        best_epoch = config.TRAIN.START_EPOCH
        no_improvement_epochs = 0
        max_no_improvement_epochs = config.TRAIN.EARLYSTOPPING.PATIENCE
        if config.TRAIN.EARLYSTOPPING.MONITOR == "auc":
            monitor = max_auc
        else:
            monitor = min_loss

    start_time = time.time()
    for epoch in range(config.TRAIN.START_EPOCH, config.TRAIN.EPOCHS):
        data_loader_train.sampler.set_epoch(epoch)

        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch, mixup_fn, lr_scheduler, scaler)
        
        acc1, acc5, auc, loss = validate(config, data_loader_val, model, is_validation=True)
        logger.info(f"Mean Accuracy of the network on the {len(dataset_val)} validation images: {acc1:.2f}%")
        logger.info(f"Mean AUC of the network on the {len(dataset_val)} validation images: {auc:.2f}%")
        logger.info(f"Mean Loss of the network on the {len(dataset_val)} validation images: {loss:.5f}")
        
        if config.TRAIN.EARLYSTOPPING.MONITOR is not None:
            has_improved, value = early_stopping(config.TRAIN.EARLYSTOPPING.MONITOR, auc, loss, monitor)
            if has_improved:
                monitor = value
                best_epoch = epoch
                no_improvement_epochs = 0
                if dist.get_rank() == 0:
                    save_checkpoint(config, epoch, model_without_ddp, max_auc, min_loss, optimizer, lr_scheduler, logger, scaler)
            else:
                if config.TRAIN.WARMUP_EPOCHS<epoch:
                    no_improvement_epochs += 1

            if no_improvement_epochs >= max_no_improvement_epochs:
                logger.info(f"No improvement in validation {config.TRAIN.EARLYSTOPPING.MONITOR} \n"
                    f"for {max_no_improvement_epochs} epochs. Stopping training.")
                break
        else:
            if dist.get_rank() == 0 and (epoch % config.SAVE_FREQ == 0 or epoch == (config.TRAIN.EPOCHS - 1)):
                save_checkpoint(config, epoch, model_without_ddp, max_auc, min_loss, optimizer, lr_scheduler, logger, scaler)

        acc1, acc5, auc, loss = validate(config, data_loader_test, model, is_validation=False)
        logger.info(f"Mean Accuracy of the network on the {len(dataset_test)} test images: {acc1:.2f}%")
        logger.info(f"Mean AUC of the network on the {len(dataset_test)} test images: {auc:.2f}%")
        logger.info(f"Mean Loss of the network on the {len(dataset_test)} test images: {loss:.5f}")
        max_auc = max(max_auc, auc)
        min_loss = min(min_loss, loss)
        logger.info(f'Test Max mean AUC: {max_auc:.2f}%')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('Training time {}'.format(total_time_str))


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch, mixup_fn, lr_scheduler, scaler):
    model.train()
    optimizer.zero_grad()

    num_steps = len(data_loader)
    batch_time = AverageMeter()
    loss_meter = AverageMeter()
    norm_meter = AverageMeter()

    start = time.time()
    end = time.time()
    grad_norm = 0.0
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        for i in range(len(targets)):
            targets[i] = targets[i].cuda(non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)   #todo iterate on targets

        if config.TRAIN.ACCUMULATION_STEPS > 1:
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.AMP_OPT_LEVEL):
                outputs = model(samples)
                loss = criterion(outputs[0], targets[0])
                for i in range(1, len(targets)):
                    loss += criterion(outputs[i], targets[i])
                loss = loss / config.TRAIN.ACCUMULATION_STEPS
            # Accumulates scaled gradients.
            scaler.scale(loss).backward()
            
            if (idx + 1) % config.TRAIN.ACCUMULATION_STEPS == 0:
                scaler.unscale_(optimizer)
                if config.TRAIN.CLIP_GRAD:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
                else:
                    grad_norm = get_grad_norm(model.parameters())
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if lr_scheduler is not None:
                    lr_scheduler.step_update(epoch * num_steps + idx)
        else:
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=config.AMP_OPT_LEVEL):
                outputs = model(samples)
                loss = criterion(outputs[0], targets[0])
                for i in range(1, len(targets)):
                    loss += criterion(outputs[i], targets[i])
            # Accumulates scaled gradients.
            scaler.scale(loss).backward()
            if config.TRAIN.CLIP_GRAD:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
            else:
                grad_norm = get_grad_norm(model.parameters())
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            if lr_scheduler is not None:
                lr_scheduler.step_update(epoch * num_steps + idx)

        torch.cuda.synchronize()

        loss_meter.update(loss.item(), targets[0].size(0))
        norm_meter.update(grad_norm)
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % config.PRINT_FREQ == 0:
            lr = optimizer.param_groups[0]['lr']
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            etas = batch_time.avg * (num_steps - idx)
            logger.info(
                f'Train: [{epoch}/{config.TRAIN.EPOCHS}][{idx}/{num_steps}]\t'
                f'eta {datetime.timedelta(seconds=int(etas))} lr {lr:.6f}\t'
                f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
                f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
                f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
                f'mem {memory_used:.0f}MB')
    lr = optimizer.param_groups[0]['lr']
    memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
    logger.info(
        f'Train: [{epoch}/{config.TRAIN.EPOCHS}]\t'
        f'lr {lr:.6f}\t'
        f'time {batch_time.val:.4f} ({batch_time.avg:.4f})\t'
        f'loss {loss_meter.val:.4f} ({loss_meter.avg:.4f})\t'
        f'grad_norm {norm_meter.val:.4f} ({norm_meter.avg:.4f})\t'
        f'mem {memory_used:.0f}MB')
    epoch_time = time.time() - start
    logger.info(f"EPOCH {epoch} training takes {datetime.timedelta(seconds=int(epoch_time))}")


@torch.no_grad()
def validate(config, data_loader, model, is_validation):
    valid_or_test = "Validation" if is_validation else "Test"
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()

    batch_time = [AverageMeter() for _ in range(14)]
    loss_meter = [AverageMeter() for _ in range(14)]
    acc1_meter = [AverageMeter() for _ in range(14)]
    acc5_meter = [AverageMeter() for _ in range(14)]

    acc1s = []
    acc5s = []
    losses = []
    aucs = []

    end = time.time()
    all_preds = [[] for _ in range(14)]
    all_label = [[] for _ in range(14)]
    for idx, (images, target) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        for i in range(len(target)):
            target[i] = target[i].cuda(non_blocking=True)

        # compute output
        output = model(images)

        for i in range(len(target)):
            # measure accuracy and record loss
            loss = criterion(output[i], target[i])
            # acc1, acc5 = accuracy(output, target, topk=(1, 5)) #https://huggingface.co/spaces/Roll20/pet_score/blob/3653888366407445408f2bfa8c68d6cdbdd4cba6/lib/timm/utils/metrics.py
            acc1 = accuracy(output[i], target[i], topk=(1,))
            acc1 = torch.Tensor(acc1).to(device='cuda')
            acc1 = reduce_tensor(acc1)
            # acc5 = reduce_tensor(acc5)
            loss = reduce_tensor(loss)

            loss_meter[i].update(loss.item(), target[i].size(0))
            acc1_meter[i].update(acc1.item(), target[i].size(0))
            # acc5_meter.update(acc5.item(), target.size(0))

            # auc
            preds = F.softmax(output[i], dim=1)
            if len(all_preds[i]) == 0:
                all_preds[i].append(preds.detach().cpu().numpy())
                all_label[i].append(target[i].detach().cpu().numpy())
            else:
                all_preds[i][0] = np.append(
                    all_preds[i][0], preds.detach().cpu().numpy(), axis=0
                )
                all_label[i][0] = np.append(
                    all_label[i][0], target[i].detach().cpu().numpy(), axis=0
                )

            # measure elapsed time
            batch_time[i].update(time.time() - end)
            end = time.time()

            if idx % config.PRINT_FREQ == 0:
                memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
                logger.info(
                    f'{valid_or_test}: [{idx}/{len(data_loader)}]\t'
                    f'Time {batch_time[i].val:.3f} ({batch_time[i].avg:.3f})\t'
                    f'Loss {loss_meter[i].val:.4f} ({loss_meter[i].avg:.4f})\t'
                    f'Acc@1 {acc1_meter[i].val:.3f} ({acc1_meter[i].avg:.3f})\t'
                    # f'Acc@5 {acc5_meter[i].val:.3f} ({acc5_meter[i].avg:.3f})\t'
                    f'Mem {memory_used:.0f}MB\t'
                    f'Class {i}')

    for i in range(14):
        # auc
        all_preds[i], all_label[i] = all_preds[i][0], all_label[i][0]
        auc = roc_auc_score(all_label[i], all_preds[i][:, 1], multi_class='ovr')
        # logger.info("Valid AUC: %2.5f" % auc)
        logger.info(f' * Acc@1 {acc1_meter[i].avg:.3f}\t'
                    f'Acc@5 {acc5_meter[i].avg:.3f}\t'
                    f'{valid_or_test} AUC {auc:.5f}\t'
                    f'Class {i}')

        acc1s.append(acc1_meter[i].avg)
        acc5s.append(acc5_meter[i].avg)
        losses.append(loss_meter[i].avg)
        aucs.append(auc)

    from statistics import mean
    logger.info(f'{valid_or_test} MEAN AUC: {mean(aucs):.5f}')

    return mean(acc1s), mean(acc5s), mean(aucs), mean(losses)


@torch.no_grad()
def throughput(data_loader, model, logger):
    model.eval()

    for idx, (images, _) in enumerate(data_loader):
        images = images.cuda(non_blocking=True)
        batch_size = images.shape[0]
        for i in range(50):
            model(images)
        torch.cuda.synchronize()
        logger.info(f"throughput averaged with 30 times")
        tic1 = time.time()
        for i in range(30):
            model(images)
        torch.cuda.synchronize()
        tic2 = time.time()
        logger.info(f"batch_size {batch_size} throughput {30 * batch_size / (tic2 - tic1)}")
        return

@torch.no_grad()
def flops(model, logger, config):
    model.eval()
    #TODO: change number of channels if channel data is different to 3
    input_tensor = torch.randn(1, 3, config.DATA.IMG_SIZE, config.DATA.IMG_SIZE).cuda()
    flops, params = thop.profile(model, inputs=(input_tensor,))
    logger.info(f"Number of GFLOPs getting with thop: {flops / 1e9}")
    logger.info(f"Number of params getting with thop: {params}")
    return

def early_stopping(monitor, monitor_auc, monitor_loss,  monitor_previus_value):
    if monitor == 'auc':
        if monitor_auc > monitor_previus_value:
            return True, monitor_auc
    if monitor == 'loss':
        if monitor_loss < monitor_previus_value:
            return True, monitor_loss
    return False, None

if __name__ == '__main__':
    _, config = parse_option()

    # if config.AMP_OPT_LEVEL:
    #     assert amp is not None, "amp not installed!"

    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        print(f"RANK and WORLD_SIZE in environ: {rank}/{world_size}")
    else:
        rank = -1
        world_size = -1
    torch.cuda.set_device(config.LOCAL_RANK)
    torch.distributed.init_process_group(backend='nccl', init_method='env://', world_size=world_size, rank=rank)
    torch.distributed.barrier()

    seed = config.SEED + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # linear scale the learning rate according to total batch size, may not be optimal
    linear_scaled_lr = config.TRAIN.BASE_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_warmup_lr = config.TRAIN.WARMUP_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    linear_scaled_min_lr = config.TRAIN.MIN_LR * config.DATA.BATCH_SIZE * dist.get_world_size() / 512.0
    # gradient accumulation also need to scale the learning rate
    if config.TRAIN.ACCUMULATION_STEPS > 1:
        linear_scaled_lr = linear_scaled_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_warmup_lr = linear_scaled_warmup_lr * config.TRAIN.ACCUMULATION_STEPS
        linear_scaled_min_lr = linear_scaled_min_lr * config.TRAIN.ACCUMULATION_STEPS
    config.defrost()
    config.TRAIN.BASE_LR = linear_scaled_lr
    config.TRAIN.WARMUP_LR = linear_scaled_warmup_lr
    config.TRAIN.MIN_LR = linear_scaled_min_lr
    config.freeze()

    os.makedirs(config.OUTPUT, exist_ok=True)
    logger = create_logger(output_dir=config.OUTPUT, dist_rank=dist.get_rank(), name=f"{config.MODEL.NAME}")

    if dist.get_rank() == 0:
        path = os.path.join(config.OUTPUT, "config.json")
        with open(path, "w") as f:
            f.write(config.dump())
        logger.info(f"Full config saved to {path}")

    # print config
    logger.info(config.dump())

    main(config)


#https://pytorch.org/docs/stable/elastic/run.html

# nohup torchrun  --nproc_per_node 1 --master_port 12345 main.py \
#   --cfg configs/MAXVIT/maxvit_small_tf_224.in1k.yaml \
#   --trainset ../data/images/ --validset ../data/images/ --testset ../data/images/ \
#   --train_csv_path configs/NIH/train.csv --valid_csv_path configs/NIH/validation.csv --test_csv_path configs/NIH/test.csv \
#   --batch-size 32 --output output/ --tag paper --num_mlp_heads 3 --amp-opt-level > log.txt & disown

#COMMAND FOR TRAINING MAXVIT, SWIN, VIT
# nohup torchrun  --nproc_per_node 1 --master_port 12345 main.py \
#   --cfg configs/MAXVIT/maxvit_large_tf_384.in21k.yaml \
#   --trainset ../data/images/ --validset ../data/images/ --testset ../data/images/ \
#   --train_csv_path configs/NIH/train.csv --valid_csv_path configs/NIH/validation.csv --test_csv_path configs/NIH/test.csv \
#   --batch-size 8 --output output/ --tag paper3 --num_mlp_heads 3 --accumulation-steps 4 --amp-opt-level > log.txt & disown

# nohup torchrun  --nproc_per_node 1 --master_port 12345 main.py \
#   --cfg configs/SWIN_TIMM/swin_tiny_patch4_window7_224.ms_in1k.yaml \
#   --trainset ../data/images/ --validset ../data/images/ --testset ../data/images/ \
#   --train_csv_path configs/NIH/train.csv --valid_csv_path configs/NIH/validation.csv --test_csv_path configs/NIH/test.csv \
#   --batch-size 32 --output output/ --tag paper --num_mlp_heads 3 --accumulation-steps 8 --amp-opt-level > log.txt & disown

# nohup torchrun  --nproc_per_node 1 --master_port 12345 main.py \
#   --cfg configs/VIT_TIMM/vit_tiny_patch16_224.augreg_in21k.yaml \
#   --trainset ../data/images/ --validset ../data/images/ --testset ../data/images/ \
#   --train_csv_path configs/NIH/train.csv --valid_csv_path configs/NIH/validation.csv --test_csv_path configs/NIH/test.csv \
#   --batch-size 32 --output output/ --tag paper --num_mlp_heads 3 --amp-opt-level > log.txt & disown

#COMMAND FOR THROUGHTPUT
#At firts: pip install htop
#deactivate config.TRAIN.AUTO_RESUME and run command without amp-opt-level:
# nohup torchrun  --nproc_per_node 1 --master_port 12345 main.py \
#   --cfg configs/SWIN_TIMM/swin_base_patch4_window7_224.ms_in21k.yaml \
#   --trainset ../data/images/ --validset ../data/images/ --testset ../data/images/ \
#   --train_csv_path configs/NIH/train.csv --valid_csv_path configs/NIH/validation.csv --test_csv_path configs/NIH/test.csv \
#   --batch-size 32 --output output/ --tag paper --num_mlp_heads 3 --throughput > log.txt & disown

#CoMMAND FOR RESCALE
#FOR SWIN
# nohup torchrun  --nproc_per_node 1 --master_port 12345 main.py \
#   --cfg configs/SWIN/swin_tiny_patch4_window12_384.yaml --pretrained output/swin_tiny_patch4_window12_384/ckpt_epoch_21.pth \
#   --trainset ../data/images/ --validset ../data/images/ --testset ../data/images/ \
#   --train_csv_path configs/NIH/train.csv --valid_csv_path configs/NIH/validation.csv --test_csv_path configs/NIH/test.csv \
#   --batch-size 32 --output output/ --tag paper --num_mlp_heads 3 --accumulation-steps 8 --amp-opt-level > log.txt & disown

#FOR VIT
# nohup torchrun  --nproc_per_node 1 --master_port 12345 main.py \
#   --cfg configs/VIT/vit_small_patch16_512.yaml --pretrained output/vit_small_patch16_512/ckpt_epoch_12.pth \
#   --trainset ../data/images/ --validset ../data/images/ --testset ../data/images/ \
#   --train_csv_path configs/NIH/train.csv --valid_csv_path configs/NIH/validation.csv --test_csv_path configs/NIH/test.csv \
#   --batch-size 16 --output output/ --tag paper --num_mlp_heads 3 --accumulation-steps 2 --amp-opt-level > log.txt & disown


# nohup torchrun  --nproc_per_node 1 --master_port 12345 main.py \
#   --cfg configs/VIT/vit_small_patch16_512.yaml --resume output/vit_small_patch16_512/ckpt_epoch_12.pth \
#   --trainset ../data/images/ --validset ../data/images/ --testset ../data/images/ \
#   --train_csv_path configs/NIH/train.csv --valid_csv_path configs/NIH/validation.csv --test_csv_path configs/NIH/test.csv \
#   --batch-size 32 --output output/ --tag paper --num_mlp_heads 3 > log.txt & disown

#PID:1452597