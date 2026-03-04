"""
Training script for semantic segmentation models.
This script handles model training with various configurations and loss functions.
"""

import os
import torch
import datetime
from functools import partial
import numpy as np
import time
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim as optim
from torch.utils.data import DataLoader

from nets.LEGDeeplab import LEGDeeplab

from nets.net_training import get_lr_scheduler, set_optimizer_lr, weights_init
from utils.callbacks import EvalCallback, LossHistory
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import (download_weights, seed_everything, show_config,worker_init_fn)
from utils.utils_fit import fit_one_epoch


if __name__ == "__main__":
    # Training start time
    start_time = time.time()
    struct_time = time.localtime(start_time)  
    time_str = time.strftime("%Y-%m-%d  %H:%M:%S", struct_time)
    total_time = 0

    # Track best model performance
    best_miou = 0.0
    best_epoch = 0

    # Hardware configuration
    Cuda = True
    seed            = 11
    distributed     = False
    sync_bn         = False
    fp16            = True
    num_classes = 4    

    # Model configuration
    model_name = "unet"  
    backbone = "resnet50"  
    pretrained  = ""  
    model_path = ""

    # Dataset and training parameters
    VOCdevkit_path = 'datasets\\VOCdevkit_coral_0.8'  
    input_shape = [512, 512]  
    save_period = 20  
    save_dir = 'logs_coral\\unet_resnet50'  
    
    # Training schedule parameters
    Init_Epoch          = 0
    Freeze_Epoch        = 30
    Freeze_batch_size   = 8   
    UnFreeze_Epoch      = 200  
    Unfreeze_batch_size = 8
    Freeze_Train        = False     
    
    # Optimization parameters
    Init_lr             = 1e-4
    Min_lr              = Init_lr * 0.01
    optimizer_type      = "adam"    
    momentum            = 0.9
    weight_decay        = 0         
    lr_decay_type       = 'step'
    
    # Evaluation and loss configuration
    eval_flag           = True  
    eval_period         = 20
    dice_loss       = True
    focal_loss      = True
    boundary_loss  = True
    cls_weights     = np.ones([num_classes], np.float32)
    
    # Data loading configuration
    num_workers     = 4      
    
    # Set random seeds for reproducibility
    seed_everything(seed)    
    
    # Distributed training setup
    ngpus_per_node  = torch.cuda.device_count()
    if distributed:
        dist.init_process_group(backend="nccl")
        local_rank  = int(os.environ["LOCAL_RANK"])
        rank        = int(os.environ["RANK"])
        device      = torch.device("cuda", local_rank)
        if local_rank == 0:
            print(f"[{os.getpid()}] (rank = {rank}, local_rank = {local_rank}) training...")
            print("Gpu Device Count : ", ngpus_per_node)
    else:
        device          = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        local_rank      = 0     
        rank            = 0

    # Initialize model
    if model_name == "LEGDeeplab":
        model = LEGDeeplab(in_channels=3, num_classes=num_classes, backbone=backbone).train()
    else:
        raise ValueError(f"Unsupported model name: {model_name}")
    
    # Initialize weights if no pre-trained model is provided
    if not pretrained:  
        weights_init(model)
    
    # Load pre-trained weights if specified
    if model_path != '':
        if local_rank == 0:
            print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path, map_location=device, weights_only=True)
        load_key, no_load_key, temp_dict = [], [], {}
        
        # Filter compatible weights
        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        if local_rank == 0:
            print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
            print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))

    if local_rank == 0:
        time_str        = datetime.datetime.strftime(datetime.datetime.now(),'%Y-%m-%d-%H-%M-%S')
        log_dir         = os.path.join(save_dir, "loss_" + str(time_str))
        loss_history    = LossHistory(log_dir, model, input_shape=input_shape)
    else:
        log_dir = None
        loss_history    = None

    if fp16:
        from torch.cuda.amp import GradScaler as GradScaler
        scaler = GradScaler()
    else:
        scaler = None

    model_train     = model.train()
    if sync_bn and ngpus_per_node > 1 and distributed:
        model_train = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model_train)
    elif sync_bn:
        print("Sync_bn is not support in one gpu or not distributed.")

    if Cuda:    
        if distributed:
            model_train = model_train.cuda(local_rank)
            model_train = torch.nn.parallel.DistributedDataParallel(model_train, device_ids=[local_rank], find_unused_parameters=True)
        else:   
            model_train = torch.nn.DataParallel(model)
            cudnn.benchmark = True              
            model_train = model_train.cuda()    

    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/train.txt"),"r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "VOC2007/ImageSets/Segmentation/val.txt"),"r") as f:
        val_lines = f.readlines()

    num_train   = len(train_lines)
    num_val     = len(val_lines)
        
    if local_rank == 0:
        show_config(
            start_time=time_str,
            fp16 = fp16, model_name = model_name, VOCdevkit_path = VOCdevkit_path,pretrained = pretrained, dice_loss= dice_loss,Focal_Loss = focal_loss, boundary_loss=boundary_loss,\
            num_classes = num_classes, backbone = backbone, model_path = model_path, input_shape = input_shape, \
            Init_Epoch = Init_Epoch, Freeze_Epoch = Freeze_Epoch, UnFreeze_Epoch = UnFreeze_Epoch, Freeze_batch_size = Freeze_batch_size, Unfreeze_batch_size = Unfreeze_batch_size, Freeze_Train = Freeze_Train, \
            Init_lr = Init_lr, Min_lr = Min_lr, optimizer_type = optimizer_type, momentum = momentum, lr_decay_type = lr_decay_type, \
            save_period = save_period, save_dir = save_dir, num_workers = num_workers, num_train = num_train, num_val = num_val,
        )
    if True:
        UnFreeze_flag = False
        if Freeze_Train:
            model.freeze_backbone()
        batch_size = Freeze_batch_size if Freeze_Train else Unfreeze_batch_size
        nbs             = 16
        lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
        lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
        Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
        Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)

        optimizer = {
            'adam'  : optim.Adam(model.parameters(), Init_lr_fit, betas = (momentum, 0.999), weight_decay = weight_decay),
            'sgd'   : optim.SGD(model.parameters(), Init_lr_fit, momentum = momentum, nesterov=True, weight_decay = weight_decay),

        }[optimizer_type]
        lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
        
        epoch_step      = num_train // batch_size
        epoch_step_val  = num_val // batch_size
        
        if epoch_step == 0 or epoch_step_val == 0:
            raise ValueError("The dataset is too small to continue training. Please expand the dataset.")
        train_dataset   = UnetDataset(
            train_lines, 
            input_shape, 
            num_classes, 
            True,        
            VOCdevkit_path 
            )
        val_dataset     = UnetDataset(val_lines, input_shape, num_classes, False, VOCdevkit_path)
        if distributed:
            train_sampler   = torch.utils.data.distributed.DistributedSampler(train_dataset, shuffle=True,)
            val_sampler     = torch.utils.data.distributed.DistributedSampler(val_dataset, shuffle=False,)
            batch_size      = batch_size // ngpus_per_node
            shuffle         = False
        else:  
            train_sampler   = None
            val_sampler     = None
            shuffle         = True

        gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                    drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler, 
                                    worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
        
        if local_rank == 0:
            eval_callback   = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                            eval_flag=eval_flag, period=eval_period)
        else:
            eval_callback   = None
        
        for epoch in range(Init_Epoch, UnFreeze_Epoch):

            epoch_start = time.time() 
            if epoch >= Freeze_Epoch and not UnFreeze_flag and Freeze_Train:
                batch_size = Unfreeze_batch_size
                nbs             = 16
                lr_limit_max    = 1e-4 if optimizer_type == 'adam' else 1e-1
                lr_limit_min    = 1e-4 if optimizer_type == 'adam' else 5e-4
                Init_lr_fit     = min(max(batch_size / nbs * Init_lr, lr_limit_min), lr_limit_max)
                Min_lr_fit      = min(max(batch_size / nbs * Min_lr, lr_limit_min * 1e-2), lr_limit_max * 1e-2)
                lr_scheduler_func = get_lr_scheduler(lr_decay_type, Init_lr_fit, Min_lr_fit, UnFreeze_Epoch)
                    
                model.unfreeze_backbone()
                            
                epoch_step      = num_train // batch_size
                epoch_step_val  = num_val // batch_size

                if epoch_step == 0 or epoch_step_val == 0:
                    raise ValueError("The dataset is too small to continue training. Please expand the dataset.")

                if distributed:
                    batch_size = batch_size // ngpus_per_node

                gen             = DataLoader(train_dataset, shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True,
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=train_sampler,
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))
                gen_val         = DataLoader(val_dataset  , shuffle = shuffle, batch_size = batch_size, num_workers = num_workers, pin_memory=True, 
                                            drop_last = True, collate_fn = unet_dataset_collate, sampler=val_sampler, 
                                            worker_init_fn=partial(worker_init_fn, rank=rank, seed=seed))

                UnFreeze_flag = True

            if distributed:
                train_sampler.set_epoch(epoch)
            set_optimizer_lr(optimizer, lr_scheduler_func, epoch)
            fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch,
                    epoch_step, epoch_step_val, gen, gen_val, UnFreeze_Epoch, Cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, log_dir,local_rank,
                        )

            epoch_time = time.time() - epoch_start
            total_time += epoch_time

            if distributed:
                dist.barrier()

        if local_rank == 0:
            end_time = time.time()                 
            final_time = time.time() - start_time   
            average_time = total_time / (UnFreeze_Epoch - Init_Epoch)  

            hours = int(total_time // 3600)
            minutes = int((total_time % 3600) // 60)
            seconds = int(total_time % 60)

            log_content = f"Training Start Time: {time_str}\n" \
                      f"Training End Time: {time.strftime('%Y-%m-%d  %H:%M:%S', time.localtime(end_time))}\n" \
                      f"Total Training Time: {hours}hours {minutes}minutes {seconds}seconds"

            with open(os.path.join(save_dir, "training_time.txt"), "a") as f:
                f.write(log_content + "\n\n")

            print('\nTraining Completed:')
            print(f'Total Training Time: {final_time // 3600:.0f}hours {final_time % 3600 // 60:.0f}minutes {final_time % 60:.2f}seconds')
            print(f'Average time per epoch: {average_time:.2f}seconds')
            print(f'Actual number of training epochs: {UnFreeze_Epoch - Init_Epoch}')

            loss_history.writer.close()