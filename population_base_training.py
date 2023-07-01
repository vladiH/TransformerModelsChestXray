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
from torch import optim as optim

from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.utils import accuracy, AverageMeter

from config import get_config
from models import build_model
from data import build_loader, build_normal_loader
from lr_scheduler import build_scheduler
from optimizer import build_optimizer
from logger import create_logger
from utils import load_checkpoint, save_checkpoint, get_grad_norm, auto_resume_helper, reduce_tensor
from sklearn.metrics import roc_auc_score

# from tqdm import tqdm
#https://pytorch.org/docs/stable/notes/amp_examples.html#amp-examples
#https://pytorch.org/tutorials/recipes/recipes/amp_recipe.html
from torch.cuda.amp import autocast, GradScaler 
from functools import partial

from ray import tune
from ray.air import Checkpoint, session
from ray.tune.schedulers import PopulationBasedTraining
from ray.tune import CLIReporter

def parse_option():
    parser = argparse.ArgumentParser('MaxVit and Swin Transformer PBT', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
    args, unparsed = parser.parse_known_args()
    config = get_config(args, True)
    config.defrost()
    config.TRAIN.EPOCHS = 10
    config.DATA.NUM_WORKERS = 16
    config.NIH.train_csv_path = "../../../configs/NIH/train.csv"
    config.NIH.valid_csv_path = "../../../configs/NIH/validation.csv"
    config.NIH.test_csv_path = "../../../configs/NIH/test.csv"
    config.NIH.trainset = "../../../../data/images/"
    config.NIH.validset = "../../../../data/images/"
    config.NIH.testset = "../../../../data/images/"
    config.freeze()
    return args, config


def main(config, num_samples=3, gpus_per_trial=1):
    scheduler = PopulationBasedTraining(
        time_attr= "training_iteration",
        perturbation_interval=2,
        metric="loss",
        mode="min",
        hyperparam_mutations={
            "weight_decay": tune.uniform(0.0, 0.3),
            "base_lr": tune.loguniform(1e-4, 1e-1),
            "batch_size": [16, 32],
            "auto_augment": [2, 4, 6, 8]
        })

    pbt_config = {
        "weight_decay": tune.choice([0.0, 0.5, 0.05, 0.005]),
        "base_lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([16, 32]),
        "auto_augment": tune.choice([2, 4, 6, 8])
    }
    
    reporter = CLIReporter(
        parameter_columns={
            "weight_decay": "w_decay",
            "base_lr": "lr",
            "batch_size": "batch_size",
            "auto_augment": "auto_augment"
        },
        metric_columns=[
            "acc", "auc", "loss", "training_iteration"
        ])
    
    result = tune.run(
        partial(train_nih, config=config),
        resources_per_trial={"cpu": config.DATA.NUM_WORKERS, "gpu": gpus_per_trial},
        config=pbt_config,
        num_samples=num_samples,
        scheduler=scheduler,
        keep_checkpoints_num=3,
        checkpoint_score_attr="training_iteration",
        progress_reporter=reporter,
        local_dir="./ray_results/",
        name="tune_transformer_pbt",
    )

    best_trial = result.get_best_trial("loss", "min", "last")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation acc: {best_trial.last_result['acc']}")
    print(f"Best trial final validation auc: {best_trial.last_result['auc']}")
    print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    # best_trained_model = Net(best_trial.config["l1"], best_trial.config["l2"])
    # device = "cpu"
    # if torch.cuda.is_available():
    #     device = "cuda:0"
    #     if gpus_per_trial > 1:
    #         best_trained_model = nn.DataParallel(best_trained_model)
    # best_trained_model.to(device)

    # best_checkpoint = best_trial.checkpoint.to_air_checkpoint()
    # best_checkpoint_data = best_checkpoint.to_dict()

    # best_trained_model.load_state_dict(best_checkpoint_data["net_state_dict"])

    # test_acc = test_accuracy(best_trained_model, device)
    # print("Best trial test set accuracy: {}".format(test_acc))

def train_nih(pbt_config, config):
    config.defrost()
    config.DATA.BATCH_SIZE = pbt_config["batch_size"]
    config.AUG.AUTO_AUGMENT = "rand-m{}-mstd0.5-inc1".format(pbt_config["auto_augment"])
    config.freeze()
    dataset_train, dataset_val, dataset_test, data_loader_train, data_loader_val, data_loader_test, mixup_fn = build_normal_loader(config, percent=0.5)

    model = build_model(config)
    model.cuda()

    skip = {}
    skip_keywords = {}
    if hasattr(model, 'no_weight_decay'):
        skip = model.no_weight_decay()
    if hasattr(model, 'no_weight_decay_keywords'):
        skip_keywords = model.no_weight_decay_keywords()
    parameters = set_weight_decay(model, skip, skip_keywords)

    optimizer = optim.AdamW(parameters, eps=config.TRAIN.OPTIMIZER.EPS, betas=config.TRAIN.OPTIMIZER.BETAS,
                                lr=pbt_config["base_lr"], weight_decay=pbt_config["weight_decay"])
    

    # lr_scheduler = build_scheduler(config, optimizer, len(data_loader_train))
    criterion = torch.nn.CrossEntropyLoss()

    if session.get_checkpoint():
        checkpoint_state = session.get_checkpoint().to_dict()
        start_epoch = checkpoint_state["epoch"]
        model.load_state_dict(checkpoint_state["model_state_dict"])
        optimizer.load_state_dict(checkpoint_state["optimizer_state_dict"])
        # lr_scheduler.load_state_dict(checkpoint_state["lr_scheduler"])
    else:
        start_epoch = 0

    for epoch in range(start_epoch, config.TRAIN.EPOCHS):
        train_one_epoch(config, model, criterion, data_loader_train, optimizer, epoch)
        
        acc1, auc, loss = validate(data_loader_val, model)
        
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            # 'lr_scheduler': lr_scheduler.state_dict(),
        }
        checkpoint = Checkpoint.from_dict(checkpoint_data)

        session.report(
            {"acc":acc1, "auc": auc, "loss": loss},
            checkpoint=checkpoint,
        )

        # acc1, acc5, loss = validate(data_loader_test, model)


def train_one_epoch(config, model, criterion, data_loader, optimizer, epoch):
    model.train()
    optimizer.zero_grad()
    # num_steps = len(data_loader)
    for idx, (samples, targets) in enumerate(data_loader):
        samples = samples.cuda(non_blocking=True)
        for i in range(len(targets)):
            targets[i] = targets[i].cuda(non_blocking=True)

        outputs = model(samples)
        loss = criterion(outputs[0], targets[0])
        for i in range(1, len(targets)):
            loss += criterion(outputs[i], targets[i])
        # Accumulates scaled gradients.
        loss.backward()
        if config.TRAIN.CLIP_GRAD:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.TRAIN.CLIP_GRAD)
        else:
            get_grad_norm(model.parameters())
        optimizer.step()
        optimizer.zero_grad()
        # lr_scheduler.step_update(epoch * num_steps + idx)

        # if idx % 10 == 0: 
        # print("[%d, %5d] loss: %.3f" % (epoch + 1, idx + 1, loss.item()))

@torch.no_grad()
def validate(data_loader, model):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval()
    loss_meter = [AverageMeter() for _ in range(14)]
    acc1_meter = [AverageMeter() for _ in range(14)]
    acc5_meter = [AverageMeter() for _ in range(14)]

    acc1s = []
    acc5s = []
    losses = []
    aucs = []
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
            # acc1 = torch.Tensor(acc1).to(device='cuda')
            # acc1 = reduce_tensor(acc1)
            # # acc5 = reduce_tensor(acc5)
            # loss = reduce_tensor(loss)
        
            loss_meter[i].update(loss.item(), target[i].size(0))
            acc1_meter[i].update(acc1[0].item(), target[i].size(0))
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

    for i in range(14):
        # auc
        all_preds[i], all_label[i] = all_preds[i][0], all_label[i][0]
        auc = roc_auc_score(all_label[i], all_preds[i][:, 1], multi_class='ovr')

        acc1s.append(acc1_meter[i].avg)
        acc5s.append(acc5_meter[i].avg)
        losses.append(loss_meter[i].avg)
        aucs.append(auc)
    from statistics import mean

    return mean(acc1s), mean(aucs), mean(losses)

def set_weight_decay(model, skip_list=(), skip_keywords=()):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
                check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            # print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

if __name__ == '__main__':
    _, config = parse_option()

    torch.manual_seed(config.SEED)
    np.random.seed(config.SEED)

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda:0")  # Specify the GPU device index (0 in this example)
        print("GPU is available.")
    else:
        device = torch.device("cpu")
        print("GPU is not available. Using CPU.")

    # Set the current device
    torch.cuda.set_device(device)

    main(config, num_samples=5, gpus_per_trial=1)



# nohup python population_base_training.py \
#   --cfg configs/MAXVIT/maxvit_small_tf_224.in1k.yaml > log.txt & disown
#1103242