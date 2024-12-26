# -*- coding: utf-8 -*-

import os
os.environ['KMP_DUPLICATE_LIB_OK']='TRUE'
# 这行为了解决报错OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized.

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# 这行为了解决报错CUDA Error: device-side assert triggered

import cv2
import copy
import time
import math
import random
import argparse
import warnings
import datetime

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.backends.cudnn as cudnn

import sys
# sys.path.append(r'/mnt/yoloxredstoridp')

import numpy as np

from loguru import logger
from datetime import timedelta

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

from yolox.exp.build_rewrite import Exp
import yolox.utils.dist as comm
from yolox.utils import (MeterBuffer,     # yolox/utils/metric.py # 不加这个括号就会报错IndentationError: unexpected indent
                         occupy_mem,      # yolox/utils/metric.py
                         gpu_mem_usage,   # yolox/utils/metric.py
                         ModelEMA,        # yolox/utils/ema.py
                         is_parallel,     # yolox/utils/ema.py
                         WandbLogger,     # yolox/utils/logger.py
                         setup_logger,    # yolox/utils/logger.py
                         all_reduce_norm, # yolox/utils/all_reduce_norm.py
                         # get_model_info,  # yolox/utils/model_utils.py
                         get_local_rank,  # yolox/utils/dist.py
                         get_world_size,  # yolox/utils/dist.py
                         get_num_devices, # yolox/utils/dist.py
                         synchronize,     # yolox/utils/dist.py
                         get_rank,        # yolox/utils/dist.py
                         load_ckpt,       # yolox/utils/checkpoint.py
                         save_checkpoint, # yolox/utils/checkpoint.py
                         configure_nccl,  # yolox/utils/setup_env.py
                         configure_omp)   # yolox/utils/setup_env.py

DEFAULT_TIMEOUT = timedelta(minutes=30)

"""Cover tools/train.py, yolox/core/launch.py, yolox/core/trainer.py, yolox/data/data_prefetcher.py"""


def visualize_inputs(inps, targets):
    """api: inps must be tensor [bs, c, h, w]"""
    for i in range(inps.shape[0]):
        temp_img = copy.deepcopy(inps[i].cpu())
        temp_img = np.ascontiguousarray(temp_img.permute(1, 2, 0).numpy())
        # temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB) # if readin by PIL.Image, then comment out this line
        cv2.imwrite(os.getcwd() + '/tblogger/' + str(random.randint(1, 500)) + '_test.jpg', temp_img) # cv2.imwrite reqire [h, w, c]


class Trainer:
    def __init__(self, exp, args):
        # init function only defines some basic attr, other attrs like model, optimizer are built in before_train methods.
        self.exp = exp
        self.args = args

        # training related attr
        self.max_epoch = exp.max_epoch
        self.amp_training = args.fp16
        self.scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        self.is_distributed = get_world_size() > 1
        self.rank = get_rank()
        self.local_rank = get_local_rank()
        self.device = "cuda:{}".format(self.local_rank)
        self.use_model_ema = exp.ema
        self.save_history_ckpt = exp.save_history_ckpt

        # data/dataloader related attr
        self.data_type = torch.float16 if args.fp16 else torch.float32
        self.input_size = exp.input_size
        self.csv_path = exp.csv_path
        self.best_ap = 0

        # metric record
        self.meter = MeterBuffer(window_size=exp.print_interval)
        self.file_name = os.path.join(exp.output_dir, args.experiment_name)

        if self.rank == 0:
            os.makedirs(self.file_name, exist_ok=True)

        setup_logger(self.file_name,distributed_rank=self.rank,filename="train_log.txt",mode="a")
        
    def train(self):
        # before_train
        # logger.info("args: {}".format(self.args)) # 只恨没早点把这三个玩意注释掉
        # logger.info("exp value:\n{}".format(self.exp))
        
        # model related init
        torch.cuda.set_device(self.local_rank)
        model = self.exp.get_model()
        # logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))
        model.to(self.device)

        # solver related init
        self.optimizer = self.exp.get_optimizer(self.args.batch_size)

        # value of epoch will be set in `resume_train`
        model = self.resume_train(model)

        # data related init
        self.no_aug = self.start_epoch >= self.max_epoch - self.exp.no_aug_epochs
        
        # 提取数据: Stitcher真正进行图片拼贴的位置在MosaicDetection_Stitcher
        self.train_loader = self.exp.get_data_loader(batch_size=self.args.batch_size, is_distributed=self.is_distributed, no_aug=self.no_aug, cache_img=self.args.cache)
        # self.train_loader_stitcher = self.exp.get_data_loader_stitcher(batch_size=self.args.batch_size * 4, is_distributed=self.is_distributed, no_aug=self.no_aug,
        #                                                                cache_img=self.args.cache)
        # self.train_loader_datapool = self.exp.get_data_loader_datapool(batch_size=self.args.batch_size, is_distributed=self.is_distributed, csv_path=self.csv_path, 
        #                                                                 no_aug=self.no_aug, cache_img=self.args.cache)
        
        logger.info("init prefetcher, this might take one minute or less...")
        self.prefetcher_regular = DataPrefetcher(self.train_loader)
        # self.prefetcher_stitcher = DataPrefetcher(self.train_loader_stitcher)
        # self.prefetcher_datapool = DataPrefetcher(self.train_loader_datapool)
        
        # print("self.train_loader: ", len(self.train_loader))
        
        # max_iter means iters per epoch
        self.max_iter = len(self.train_loader)

        self.lr_scheduler = self.exp.get_lr_scheduler(self.exp.basic_lr_per_img * self.args.batch_size, self.max_iter)
        if self.args.occupy:
            occupy_mem(self.local_rank)

        if self.is_distributed:
            model = DDP(model, device_ids=[self.local_rank], broadcast_buffers=False)

        if self.use_model_ema:
            self.ema_model = ModelEMA(model, 0.9998)
            self.ema_model.updates = self.max_iter * self.start_epoch

        self.model = model
        self.model.train()

        self.evaluator = self.exp.get_evaluator(batch_size=self.args.batch_size, is_distributed=self.is_distributed)
        
        # Tensorboard and Wandb loggers
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger = SummaryWriter(os.path.join(self.file_name, "tensorboard"))
            elif self.args.logger == "wandb":
                wandb_params = dict()
                for k, v in zip(self.args.opts[0::2], self.args.opts[1::2]):
                    if k.startswith("wandb-"):
                        wandb_params.update({k.lstrip("wandb-"): v})
                self.wandb_logger = WandbLogger(config=vars(self.exp), **wandb_params)
            else:
                raise ValueError("logger must be either 'tensorboard' or 'wandb'")

        logger.info("Training start...")
        # logger.info("\n{}".format(model))
        
        ratio, scm_list, scb_list, sca_list = [], [], [], []

        # 初始化三个边距列表
        cls_list, cls_max, cls_min = [[], [], []], [0, 0, 0], [0, 0, 0]
        obj_list, obj_max, obj_min = [[], [], []], [0, 0, 0], [0, 0, 0]
        
        # train_in_iter 外层for循环
        for self.epoch in range(self.start_epoch, self.max_epoch):
            # before_epoch
            logger.info("---> start train epoch{}".format(self.epoch + 1))
            
            self.prefetcher = self.prefetcher_regular
            
            if self.epoch + 1 == self.max_epoch - self.exp.no_aug_epochs or self.no_aug:
                logger.info("--->No mosaic aug now!")
                self.train_loader.close_mosaic()
                logger.info("--->Add additional L1 loss now!")
                if self.is_distributed:
                    self.model.module.head.use_l1 = True
                else:
                    self.model.head.use_l1 = True
                self.exp.eval_interval = 1
                if not self.no_aug:
                    self.save_ckpt(ckpt_name="last_mosaic_epoch")
            
            # train_in_iter 内层for循环
            for self.iter in range(self.max_iter): # self.max_iter=len(self.train_loader),即数据集被划分成多少个批次
                # self.dynamic_controller(self, ratio_scale)
                # self.global_controller(self)
                # self.hybrid_controller(self, ratio_scale)
                # self.warmup_controller(self, ratio_scale)
                
                iter_start_time = time.time()
                
                inps, targets = self.prefetcher.next() # inps&targets均为张量,[2,3,640,640]&[2,120,5] cls, xywh
                # visualize_inputs(inps, targets)
                
                inps = inps.to(self.data_type)
                targets = targets.to(self.data_type)
                inps = inps.to(self.device)
                targets = targets.to(self.device)
                targets.requires_grad = False
                inps, targets = self.exp.preprocess(inps, targets, self.input_size)
                
                data_end_time = time.time()
                
                with torch.cuda.amp.autocast(enabled=self.amp_training):
                    outputs = self.model(inps, targets, cls_list, cls_max, cls_min, obj_list, obj_max, obj_min, self.epoch, self.max_epoch)
                
                ratio.append(1 if outputs["ratio"][0] < 0.1 else 0)
                scm_list.append(outputs["scm"])
                scb_list.append(outputs["scb"])
                sca_list.append(outputs["sca"])

                del outputs["ratio"]
                del outputs["scm"]
                del outputs["scb"]
                del outputs["sca"]

                self.optimizer.zero_grad()
                # self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                if self.use_model_ema:
                    self.ema_model.update(self.model)

                lr = self.lr_scheduler.update_lr(self.progress_in_iter + 1)
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                iter_end_time = time.time()
                
                self.meter.update(iter_time=iter_end_time - iter_start_time, data_time=data_end_time - iter_start_time, lr=lr, **outputs)

                # normalized margin的停止机制
                if len(ratio) == 100:
                    return [ratio, scm_list, scb_list, sca_list]
                
                # 设定终止机制, 5轮即停止训练
                # if (self.epoch - self.start_epoch) == 10:
                #     return 0, 0, 0, 0
                
                # after_iter log needed information
                if (self.iter + 1) % self.exp.print_interval == 0:

                    # TODO check ETA logic
                    left_iters = self.max_iter * self.max_epoch - (self.progress_in_iter + 1)
                    eta_seconds = self.meter["iter_time"].global_avg * left_iters
                    eta_str = "ETA: {}".format(datetime.timedelta(seconds=int(eta_seconds)))

                    progress_str = "epoch: {}/{}, iter: {}/{}".format(self.epoch + 1, self.max_epoch, self.iter + 1, self.max_iter)
                    loss_meter = self.meter.get_filtered_meter("loss")
                    loss_str = ", ".join(["{}: {:.1f}".format(k, v.latest) for k, v in loss_meter.items()])

                    time_meter = self.meter.get_filtered_meter("time")
                    time_str = ", ".join(["{}: {:.3f}s".format(k, v.avg) for k, v in time_meter.items()])

                    logger.info("{}, mem: {:.0f}Mb, {}, {}, lr: {:.3e}".format(progress_str, gpu_mem_usage(), time_str, loss_str, self.meter["lr"].latest) + \
                               (", size: {:d}, {}".format(self.input_size[0], eta_str))) #  + \
                               # (", mosaic: {}, stitcher:{}, datapool: {}".format(self.iter_normal, self.iter_stitcher, self.iter_datapool))) # + \
                               # (", total_ece: {:.2f}, small_ece: {:.2f}, middle_ece: {:.2f}, large_ece: {:.2f}".format(scale_ece[0], scale_ece[1], scale_ece[2], scale_ece[3])))
                               # (", s_loss: {}, m_loss: {}, l_loss: {}".format(self.loss_sum_small, self.loss_sum_medium, self.loss_sum_large)))

                    if self.rank == 0:
                        if self.args.logger == "wandb":
                            self.wandb_logger.log_metrics({k: v.latest for k, v in loss_meter.items()})
                            self.wandb_logger.log_metrics({"lr": self.meter["lr"].latest})

                    self.meter.clear_meters()

                # random resizing
                if (self.progress_in_iter + 1) % 10 == 0:
                    self.input_size = self.exp.random_resize(self.train_loader, self.epoch, self.rank, self.is_distributed)
            
            # after_epoch 内层for循环结束
            self.save_ckpt(f"epoch_{self.epoch + 1}") # ckpt_name="latest"
            
            if (self.epoch + 1) % self.exp.eval_interval == 0:
                all_reduce_norm(self.model)
                self.evaluate_and_save_model()
            
        # after_train 外层for循环结束
        logger.info("Training of experiment is done and the best AP is {:.2f}".format(self.best_ap * 100))
        if self.rank == 0:
            if self.args.logger == "wandb":
                self.wandb_logger.finish()
    
    def dynamic_controller(self, ratio_scale):
        if ratio_scale[0] < 0.1:
            self.prefetcher = self.prefetcher_stitcher
            self.iter_stitcher = self.iter_stitcher + 1
        else:
            self.prefetcher = self.prefetcher_regular
            self.iter_normal = self.iter_normal + 1

    def global_controller(self):
        if random.random() < 0.9: # self.thr_regulator()
            self.prefetcher = self.prefetcher_regular
            self.iter_normal = self.iter_normal + 1
        else:
            self.prefetcher = self.prefetcher_datapool
            self.iter_datapool = self.iter_datapool + 1
    
    def hybrid_controller(self, ratio_scale):
        if ratio_scale[0] < 0.1:
            if self.batch_flag != 0:
                self.prefetcher = self.prefetcher_regular
                self.batch_flag = 0
            else:
                if random.random() * 100 <= self.yolox_warm_cos_stitcher(0.1, self.max_epoch, self.epoch) * 1000:
                    print("initialize stitcher augment")
                    self.prefetcher = self.prefetcher_stitcher
                    self.batch_flag = 1
                else:
                    self.prefetcher = self.prefetcher_regular
                    self.batch_flag = 0
        else:
            self.prefetcher = self.prefetcher_regular
            self.batch_flag = 0

    def warmup_controller(self, ratio_scale): 
        # 分段增强, 先用致密Aug, 再用正常Aug
        if self.epoch <= -1:
            self.prefetcher = self.prefetcher_stitcher
            self.iter_stitcher = self.iter_stitcher + 1
        else: # 已经完成初始化轮次后
            # 首先检查上一轮次是否为非正常轮次, 如果不检查上一轮次是否为Stitcher, 那么很可能永远无法从DataPool中出来
            if self.batch_flag != 4:
                if ratio_scale[0] < 0.1: # 判断小目标损失值, 当小目标损失值小于阈值时启动目标池增强
                    self.prefetcher = self.prefetcher_datapool
                    self.iter_datapool = self.iter_datapool + 1
                    self.batch_flag = 3 # 目标池图片赋3号标识
                else: # 当小目标损失值大于阈值时启动正常增强
                    # Mosaic与Stitcher比例为93:7
                    if random.random() < self.thr_regulator(): # 93%的概率启动Mosaic
                        self.prefetcher = self.prefetcher_regular
                        self.iter_normal = self.iter_normal + 1
                        self.batch_flag = 0 # Mosaic图片赋0号标识
                    else: # 7%的概率启动Stitcher
                        self.prefetcher = self.prefetcher_datapool
                        self.iter_datapool = self.iter_datapool + 1
                        self.batch_flag = 4 # 四合一图片赋4号标识
            else: # 当上一轮次为非正常轮次时, 即使用Stitcher的轮次
                # 设定Mosaic与Stitcher比例为93:7
                if random.random() < self.thr_regulator(): # 93%的概率启动Mosaic
                    self.prefetcher = self.prefetcher_regular
                    self.iter_normal = self.iter_normal + 1
                    self.batch_flag = 0 # Mosaic图片赋0号标识
                else: # 7%的概率启动Stitcher
                    self.prefetcher = self.prefetcher_datapool
                    self.iter_datapool = self.iter_datapool + 1
                    self.batch_flag = 4 # 四合一图片赋4号标识"""

    def initialvalue_search(self):
        return 95.00 / 100
    
    def thr_regulator(self):
        return (93.00 + 1.47 * math.log(self.epoch + 1, 10)) / 100
        # return 1 - self.yolox_warm_cos_stitcher(0.07, self.max_epoch, self.epoch)
        
    def yolox_warm_cos_stitcher(self, prob, total_epoch, current_epoch):
        prob = 0.5 * prob * (1.0 + math.cos(math.pi * current_epoch / total_epoch))
        return prob
    
    @property
    def progress_in_iter(self):
        return self.epoch * self.max_iter + self.iter

    def resume_train(self, model):
        if self.args.resume:
            logger.info("resume training")
            if self.args.ckpt is None:
                ckpt_file = os.path.join(self.file_name, "latest" + "_ckpt.pth")
            else:
                ckpt_file = self.args.ckpt
            ckpt = torch.load(ckpt_file, map_location=self.device)
            
            # resume the model/optimizer state dict
            model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])
            self.best_ap = ckpt.pop("best_ap", 0)
            
            # resume the training states variables
            start_epoch = (self.args.start_epoch - 1 if self.args.start_epoch is not None else ckpt["start_epoch"])
            self.start_epoch = start_epoch
            logger.info("loaded checkpoint '{}' (epoch {})".format(self.args.resume, self.start_epoch))  # noqa
        else:
            if self.args.ckpt is not None:
                logger.info("loading checkpoint for fine tuning")
                ckpt_file = self.args.ckpt
                ckpt = torch.load(ckpt_file, map_location=self.device)["model"]
                model = load_ckpt(model, ckpt)
            self.start_epoch = 0

        return model

    def ratio_searching_evaluate(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module
        ap50_95, ap50, summary = self.evaluator.evaluate(evalmodel, self.is_distributed, half=False)
         
        self.model.train()
         
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("searching/AP50", ap50, self.epoch + 1)
        synchronize()

    def evaluate_and_save_model(self):
        if self.use_model_ema:
            evalmodel = self.ema_model.ema
        else:
            evalmodel = self.model
            if is_parallel(evalmodel):
                evalmodel = evalmodel.module

        # ap50_95, ap50, summary = self.exp.eval(evalmodel, self.evaluator, self.is_distributed)
        ap50_95, ap50, summary = self.evaluator.evaluate(evalmodel, self.is_distributed, half=False)
        update_best_ckpt = ap50 > self.best_ap
        self.best_ap = max(self.best_ap, ap50)

        self.model.train()
        if self.rank == 0:
            if self.args.logger == "tensorboard":
                self.tblogger.add_scalar("val/COCOAP50", ap50, self.epoch + 1)
                self.tblogger.add_scalar("val/COCOAP50_95", ap50_95, self.epoch + 1)
            if self.args.logger == "wandb":
                self.wandb_logger.log_metrics({"val/COCOAP50": ap50,
                                               "val/COCOAP50_95": ap50_95,
                                               "epoch": self.epoch + 1})
            logger.info("\n" + summary)
        synchronize()

        self.save_ckpt("last_epoch", update_best_ckpt)
        if self.save_history_ckpt:
            self.save_ckpt(f"epoch_{self.epoch + 1}")

    def save_ckpt(self, ckpt_name, update_best_ckpt=False):
        if self.rank == 0:
            save_model = self.ema_model.ema if self.use_model_ema else self.model
            logger.info("Save weights to {}".format(self.file_name))
            ckpt_state = {"start_epoch": self.epoch + 1,
                          "model": save_model.state_dict(),
                          "optimizer": self.optimizer.state_dict(),
                          "best_ap": self.best_ap}
            save_checkpoint(ckpt_state,update_best_ckpt,self.file_name,ckpt_name)

            if self.args.logger == "wandb":
                self.wandb_logger.save_checkpoint(self.file_name, ckpt_name, update_best_ckpt)
    
    def loss_difference(self, loss, loss_small_total, loss_midle_total, loss_large_total, 
                        loss_small_iou, loss_midle_iou, loss_large_iou, loss_iou, 
                        loss_small_cls, loss_midle_cls, loss_large_cls, loss_cls, 
                        loss_small_obj, loss_midle_obj, loss_large_obj, loss_obj):
        # 判断单位时间内不同尺度目标的优化差异(希望证明大目标的优化差异总是强于小目标)
        loss_small_total[self.epoch * self.max_iter + self.iter] = self.scaler.scale(5.0 * loss_iou[0] + loss_obj[0] + loss_cls[0])
        loss_midle_total[self.epoch * self.max_iter + self.iter] = self.scaler.scale(5.0 * loss_iou[1] + loss_obj[1] + loss_cls[1])
        loss_large_total[self.epoch * self.max_iter + self.iter] = self.scaler.scale(5.0 * loss_iou[2] + loss_obj[2] + loss_cls[2])

        loss_small_iou[self.epoch * self.max_iter + self.iter] = self.scaler.scale(loss_iou[0])
        loss_midle_iou[self.epoch * self.max_iter + self.iter] = self.scaler.scale(loss_iou[1])
        loss_large_iou[self.epoch * self.max_iter + self.iter] = self.scaler.scale(loss_iou[2])

        loss_small_cls[self.epoch * self.max_iter + self.iter] = self.scaler.scale(loss_cls[0])
        loss_midle_cls[self.epoch * self.max_iter + self.iter] = self.scaler.scale(loss_cls[1])
        loss_large_cls[self.epoch * self.max_iter + self.iter] = self.scaler.scale(loss_cls[2])

        loss_small_obj[self.epoch * self.max_iter + self.iter] = self.scaler.scale(loss_obj[0])
        loss_midle_obj[self.epoch * self.max_iter + self.iter] = self.scaler.scale(loss_obj[1])
        loss_large_obj[self.epoch * self.max_iter + self.iter] = self.scaler.scale(loss_obj[2])

        if len(loss_small_total) > self.exp.print_interval:
            del loss_small_total[sorted(list(loss_small_total.keys()))[0]]
            del loss_midle_total[sorted(list(loss_midle_total.keys()))[0]]
            del loss_large_total[sorted(list(loss_large_total.keys()))[0]]

            del loss_small_iou[sorted(list(loss_small_iou.keys()))[0]]
            del loss_midle_iou[sorted(list(loss_midle_iou.keys()))[0]]
            del loss_large_iou[sorted(list(loss_large_iou.keys()))[0]]

            del loss_small_cls[sorted(list(loss_small_cls.keys()))[0]]
            del loss_midle_cls[sorted(list(loss_midle_cls.keys()))[0]]
            del loss_large_cls[sorted(list(loss_large_cls.keys()))[0]]

            del loss_small_obj[sorted(list(loss_small_obj.keys()))[0]]
            del loss_midle_obj[sorted(list(loss_midle_obj.keys()))[0]]
            del loss_large_obj[sorted(list(loss_large_obj.keys()))[0]]

        # 展示由于随机训练所以每个批次的尺度差异都不同, 而学习率会缩小这种差异
        if self.rank == 0 and self.args.logger == 'tensorboard':
            self.tblogger.add_scalar("train/every_batch_unscaled_total_loss", loss, (self.epoch) * self.max_iter + self.iter) # 这是每个样本具有的损失值, 因此非常小, 只有不到10
            self.tblogger.add_scalar("train/every_batch_scaled_total_loss_dif", loss_large_total[self.epoch * self.max_iter + self.iter] - loss_small_total[self.epoch * self.max_iter + self.iter], (self.epoch) * self.max_iter + self.iter)
            self.tblogger.add_scalar("train/every_batch_scaled_iou_loss_dif", loss_large_iou[self.epoch * self.max_iter + self.iter] - loss_small_iou[self.epoch * self.max_iter + self.iter], (self.epoch) * self.max_iter + self.iter)
            self.tblogger.add_scalar("train/every_batch_scaled_cls_loss_dif", loss_large_cls[self.epoch * self.max_iter + self.iter] - loss_small_cls[self.epoch * self.max_iter + self.iter], (self.epoch) * self.max_iter + self.iter)
            self.tblogger.add_scalar("train/every_batch_scaled_obj_loss_dif", loss_large_obj[self.epoch * self.max_iter + self.iter] - loss_small_obj[self.epoch * self.max_iter + self.iter], (self.epoch) * self.max_iter + self.iter)
            # print(f"{self.epoch * self.max_iter + self.iter} batch scaled total loss value difference between large & small objects: {loss_large_total[self.epoch * self.max_iter + self.iter] - loss_small_total[self.epoch * self.max_iter + self.iter]}")
            # print(f"{self.epoch * self.max_iter + self.iter} batch scaled iou loss value difference between large & small objects: {loss_large_iou[self.epoch * self.max_iter + self.iter] - loss_small_iou[self.epoch * self.max_iter + self.iter]}")
            # print(f"{self.epoch * self.max_iter + self.iter} batch scaled cls loss value difference between large & small objects: {loss_large_cls[self.epoch * self.max_iter + self.iter] - loss_small_cls[self.epoch * self.max_iter + self.iter]}")
            # print(f"{self.epoch * self.max_iter + self.iter} batch scaled obj loss value difference between large & small objects: {loss_large_obj[self.epoch * self.max_iter + self.iter] - loss_small_obj[self.epoch * self.max_iter + self.iter]}")
            # print("-" * 20)

        # 展示T时间内大目标用于反向传播的损失值总大于小目标(用于convincing审稿人)
        if len(loss_small_total) == self.exp.print_interval:
            if self.rank == 0 and self.args.logger == 'tensorboard':
                self.tblogger.add_scalar("train/Nbatches_scaled_loss_dif_total", sum(list(loss_large_total.values())) - sum(list(loss_small_total.values())), (self.epoch) * self.max_iter + self.iter)
                self.tblogger.add_scalar("train/Nbatches_scaled_loss_dif_iou", sum(list(loss_large_iou.values())) - sum(list(loss_small_iou.values())), (self.epoch) * self.max_iter + self.iter)
                self.tblogger.add_scalar("train/Nbatches_scaled_loss_dif_cls", sum(list(loss_large_cls.values())) - sum(list(loss_small_cls.values())), (self.epoch) * self.max_iter + self.iter)
                self.tblogger.add_scalar("train/Nbatches_scaled_loss_dif_obj", sum(list(loss_large_obj.values())) - sum(list(loss_small_obj.values())), (self.epoch) * self.max_iter + self.iter)
    
        # if (self.iter + 1) % self.exp.print_interval == 0:
        #     if len(loss_small_total) == self.exp.print_interval:
        #         print(f"N batches scaled total loss value difference between large & small objects: {sum(list(loss_large_total.values())) - sum(list(loss_small_total.values()))}")
        #         print(f"N batches scaled iou loss value difference between large & small objects: {sum(list(loss_large_iou.values())) - sum(list(loss_small_iou.values()))}")
        #         print(f"N batches scaled cls loss value difference between large & small objects: {sum(list(loss_large_cls.values())) - sum(list(loss_small_cls.values()))}")
        #         print(f"N batches scaled obj loss value difference between large & small objects: {sum(list(loss_large_obj.values())) - sum(list(loss_small_obj.values()))}")
    
        return loss_small_total, loss_midle_total, loss_large_total, \
               loss_small_iou, loss_midle_iou, loss_large_iou, \
               loss_small_cls, loss_midle_cls, loss_large_cls, \
               loss_small_obj, loss_midle_obj, loss_large_obj
    
    def whole_network_grad_sim(self):
        # 初始化当前批次的全局梯度收集器以备填充梯度
        grad_dims = []
        for name, param in self.model.named_parameters():
            grad_dims.append(param.data.numel())
        grad_tensor = torch.Tensor(sum(grad_dims))
        grad_tensor.fill_(0.0)

        # 从网络中获取每层的梯度和参数并组成一个张量
        counter = 0
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if counter == 0:
                    begin_point = 0
                else:
                    begin_point = sum(grad_dims[:counter])
                end_point = sum(grad_dims[:counter + 1])
                grad_tensor[begin_point: end_point].copy_(param.grad.data.view(-1))
            counter += 1
        grad_tensor = grad_tensor.unsqueeze(0)
        
        # 计算上一批次与当前批次的余弦相似度
        if isinstance(self.grad_last, float):
            self.grad_last = copy.deepcopy(grad_tensor)
        else:
            self.cos_sim = cosine_similarity(self.grad_last, grad_tensor).item() # get_cosine_similarity(self.grad_last, grad_tensor).item()
            self.grad_last = copy.deepcopy(grad_tensor)
            print(f"the cosine similarity and theta of last and current gradient: {self.cos_sim}, {math.degrees(math.acos(self.cos_sim))}")
        
        # 使用tensorboard登记余弦相似度
        if self.rank == 0 and self.args.logger == 'tensorboard':
            self.tblogger.add_scalar("train/whole_net_grad_sim", math.degrees(math.acos(self.cos_sim)), (self.epoch) * self.max_iter + self.iter)
        
        # return math.degrees(math.acos(self.cos_sim))
    
    def layer_by_layer_grad_sim(self):
        # 第一个batch只记录每层梯度
        if isinstance(self.grad_last, float):
            self.grad_last = {}
            for name, param in self.model.named_parameters():
                self.grad_last[name] = param.grad.data.view(-1)
                self.grad_last[name] = self.grad_last[name].unsqueeze(0)
        
        # 其他batch开始计算每层相似度
        else:
            grad_sim, grad_sim_mean, counter = {}, 0.0, 0
            for name, param in self.model.named_parameters():
                if param.grad is not None and "conv.weight" in name:
                    grad_current = copy.deepcopy(param.grad.data.view(-1)).unsqueeze(0)
                    grad_sim[name] = cosine_similarity(self.grad_last[name], grad_current).item()
                    self.grad_last[name] = grad_current
                    grad_sim_mean += math.degrees(math.acos(grad_sim[name]))
                    counter += 1
            print(f"the mean gradient similarity: {grad_sim_mean / counter}")
    
    def intra_batch_grad_sim(self):
        inps_small = copy.deepcopy(inps).detach()
        inps_large = copy.deepcopy(inps).detach()
        inps_small.requires_grad = True
        inps_large.requires_grad = True
        small_gradients = torch.autograd.grad(outputs=self.scaler.scale(loss_iou[0]), inputs=inps_small, create_graph=True, allow_unused=True, retain_graph=True, only_inputs=True)
        large_gradients = torch.autograd.grad(outputs=self.scaler.scale(loss_iou[2]), inputs=inps_large, create_graph=True, allow_unused=True, retain_graph=True, only_inputs=True)
        print(f"the small and large object gradient: {small_gradients}, {large_gradients}")
        self.cos_sim = get_cosine_similarity(small_gradients, large_gradients).item()
        print(f"the cosine similarity and theta of large and small gradient: {self.cos_sim}, {math.degrees(math.acos(self.cos_sim))}")
    
    def batch_scale_margin(self, margin, margin_small, margin_midle, margin_large, margin_total):
        """
        smooth batch level margin, making multiple notes in training
        """
        # 记录每个batch的各尺度margin
        margin_small.append(margin[0])
        margin_midle.append(margin[1])
        margin_large.append(margin[2])
        margin_total.append(margin[3])

        # 移除过多的元素, 将元素控制在N个
        if len(margin_small) > self.exp.print_interval * 100:
            margin_small.pop(0)
            margin_midle.pop(0)
            margin_large.pop(0)
            margin_total.pop(0)

        # 向tensorboard更新平滑后margin
        if len(margin_small) == self.exp.print_interval * 100:
            if self.rank == 0 and self.args.logger == 'tensorboard':
                self.tblogger.add_scalar("train/batch_small_margin", sum(margin_small) / len(margin_small), (self.epoch) * self.max_iter + self.iter)
                self.tblogger.add_scalar("train/batch_midle_margin", sum(margin_midle) / len(margin_midle), (self.epoch) * self.max_iter + self.iter)
                self.tblogger.add_scalar("train/batch_large_margin", sum(margin_large) / len(margin_large), (self.epoch) * self.max_iter + self.iter)
                self.tblogger.add_scalar("train/batch_total_margin", sum(margin_total) / len(margin_total), (self.epoch) * self.max_iter + self.iter)
        
        return margin_small, margin_midle, margin_large, margin_total

    def normalize_margin(self, targets, margin, margin_small, margin_midle, margin_large, margin_total):
        area_small, area_midle, area_large, area_total = [], [], [], []
        for i in range(len(targets)):
            for j in range(len(targets[i])):
                area = targets[i][j][3].item() * targets[i][j][4].item()
                if 0 < area <= 32 * 32:
                    area_small.append(area)
                if 32 * 32 < area <= 96 * 96:
                    area_midle.append(area)
                if area > 96 * 96:
                    area_large.append(area)
                if 0 < area:
                    area_total.append(area)
        
        if len(area_small) != 0:
            margin_small.append(margin[0] / (sum(area_small) / len(area_small)))
        if len(area_midle) != 0:
            margin_midle.append(margin[1] / (sum(area_midle) / len(area_midle)))
        if len(area_large) != 0:
            margin_large.append(margin[2] / (sum(area_large) / len(area_large)))
        if len(area_total) != 0:
            margin_total.append(margin[3] / (sum(area_total) / len(area_total)))

        return margin_small, margin_midle, margin_large, margin_total

    def loss_distributer(self, loss_curve, loss_scale):
        for i in range(len(loss_scale)):
            dict_key = int(list(loss_scale[i].keys())[0])
            dict_value = list(loss_scale[i].values())[0]
            loss_curve[dict_key].append(dict_value)
        return loss_curve


# 两种余弦相似度计算方法
def cosine_similarity(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)

    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    sim= torch.mm(x1, x2.t())/(w1 * w2.t()) #, w1  # .clamp(min=eps), 1/cosinesim

    return sim

def batch_dot(a,b):
    return torch.sum(a.view([a.size(0),-1])*b.view([b.size(0),-1]), dim=1)

def stabilize_variable(v):
    v_sign = torch.sign(v.detach())
    v_sign[v_sign==0] = 1
    return v+v_sign*1e-15

def get_cosine_similarity(a,b):
    a = stabilize_variable(a)
    b = stabilize_variable(b)
    a = a.view([a.size(0),-1])
    b = b.view([b.size(0),-1])
    return batch_dot(a,b)/torch.norm(torch.abs(a), dim=1)/torch.norm(torch.abs(b), dim=1)


class DataPrefetcher:
    """
    DataPrefetcher is inspired by code of following file:
    https://github.com/NVIDIA/apex/blob/master/examples/imagenet/main_amp.py
    It could speedup your pytorch dataloader. For more information, please check
    https://github.com/NVIDIA/apex/issues/304#issuecomment-493562789.
    """

    def __init__(self, loader): # 输入的loader是yolox_base.py的get_data_loader函数
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.input_cuda = self._input_cuda_for_image
        self.record_stream = DataPrefetcher._record_stream_for_image
        self.preload()

    def preload(self):
        try:
            self.next_input, self.next_target, _, _ = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return

        with torch.cuda.stream(self.stream):
            self.input_cuda()
            self.next_target = self.next_target.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        target = self.next_target
        if input is not None:
            self.record_stream(input)
        if target is not None:
            target.record_stream(torch.cuda.current_stream())
        self.preload()
        return input, target

    def _input_cuda_for_image(self):
        self.next_input = self.next_input.cuda(non_blocking=True)

    @staticmethod
    def _record_stream_for_image(input):
        input.record_stream(torch.cuda.current_stream())


def make_parser():
    parser = argparse.ArgumentParser("YOLOX train parser")

    parser.add_argument("-expn", "--experiment-name", type=str, default=None)
    parser.add_argument("-n", "--name", type=str, default=None, help="model name")
    
    # batch size
    parser.add_argument("-b", "--batch-size", type=int, default=16, help="batch size")
    
    # network params
    parser.add_argument("-d", "--devices",       default=None,   type=int,                           help="device for training")
    parser.add_argument("-f", "--exp_file",      default=None,   type=str,                           help="plz input your experiment description file")

    # pre-train
    parser.add_argument("--resume",              default=True,   action="store_true",                help="resume training")
    parser.add_argument("-c", "--ckpt",          default=None,   type=str,                           help="checkpoint file") # "./YOLOX_outputs/Baseline_S_78.07/epoch_100_ckpt.pth"
    parser.add_argument("-e", "--start_epoch",   default=None,   type=int,                           help="resume training start epoch")

    # distribute training
    parser.add_argument("--num_machines",        default=1,      type=int,                           help="num of node for training")
    parser.add_argument("--machine_rank",        default=0,      type=int,                           help="node rank for multi-node training")
    parser.add_argument("--dist-backend",        default="nccl", type=str,                           help="distributed backend")
    parser.add_argument("--dist-url",            default=None,   type=str,                           help="url used to set up distributed training")

    # other params
    parser.add_argument("--fp16",                default=False,  dest="fp16",   action="store_true", help="Adopting mix precision training.")
    parser.add_argument("--cache",               default=False,  dest="cache",  action="store_true", help="Caching imgs to RAM for fast training.")
    parser.add_argument("-o","--occupy",         default=False,  dest="occupy", action="store_true", help="occupy GPU memory first for training.")
    parser.add_argument("-l","--logger",                         type=str,                           help="Logger to be used for metrics",default="tensorboard")
    parser.add_argument("opts",                  default=None,   nargs=argparse.REMAINDER,           help="Modify config options using the command-line")
    return parser


@logger.catch
def main(exp, args):
    # 设定每次使用相同的随机数种子使得CNN每次初始化一致,每次训练结果一致
    if exp.seed is not None:
        random.seed(exp.seed)
        torch.manual_seed(exp.seed) # 为CPU设置固定的随机数种子
        cudnn.deterministic = True
        warnings.warn("You have chosen to seed training. This will turn on the CUDNN deterministic setting, "
                      "which can slow down your training considerably! You may see unexpected behavior "
                      "when restarting from checkpoints.")

    # set environment variables for distributed training
    configure_nccl()
    configure_omp()
    cudnn.benchmark = True

    trainer = Trainer(exp, args)
    margin_list = trainer.train()
    return margin_list


def _distributed_worker(local_rank,main_func,world_size,num_gpus_per_machine,machine_rank,backend,dist_url,args,timeout=DEFAULT_TIMEOUT,):
    assert (torch.cuda.is_available()), "cuda is not available. Please check your installation."
    global_rank = machine_rank * num_gpus_per_machine + local_rank
    logger.info("Rank {} initialization finished.".format(global_rank))
    try:
        dist.init_process_group(backend=backend, init_method=dist_url, world_size=world_size, rank=global_rank, timeout=timeout)
    except Exception:
        logger.error("Process group URL: {}".format(dist_url))
        raise

    # Setup the local process group (which contains ranks within the same machine)
    assert comm._LOCAL_PROCESS_GROUP is None
    num_machines = world_size // num_gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(i * num_gpus_per_machine, (i + 1) * num_gpus_per_machine))
        pg = dist.new_group(ranks_on_i)
        if i == machine_rank:
            comm._LOCAL_PROCESS_GROUP = pg

    comm.synchronize()

    assert num_gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    main_func(*args)


# if __name__ == "__main__":
def yoloxdstre(ckpt_path, resume):
    args = make_parser().parse_args() # parser模块实例化

    # addition codes
    if resume:
        args.ckpt = ckpt_path
    args.resume = resume

    exp = Exp() # 参数控制类负责存储参数,调用网络架构&训练集加载器&测试集加载器&验证集加载器&优化器&学习率调整器,包含预处理等其他函数
    exp.merge(args.opts)

    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    num_gpu = get_num_devices() if args.devices is None else args.devices
    
    assert num_gpu <= get_num_devices()
    
    # num_gpu = 0 # 当使用cpu进行训练或测试时将gpu数量设为0

    dist_url = "auto" if args.dist_url is None else args.dist_url # 并行训练起始地址

    world_size = args.num_machines * num_gpu # 计算GPU的总数量,机器数量×每台机器的GPU数量
    
    if world_size > 1: # 有卡就并行训练
        if dist_url == "auto":
            assert (args.num_machines == 1), "dist_url=auto cannot work with distributed training."
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("", 0))
            port = sock.getsockname()[1]
            sock.close()
            dist_url = f"tcp://127.0.0.1:{port}"

        start_method = "spawn"
        cache = vars(args[1]).get("cache", False)

        # To use numpy memmap for caching image into RAM, we have to use fork method
        if cache:
            assert sys.platform != "win32", ("As Windows platform doesn't support fork method, "
                                             "do not add --cache in your training command.")
            start_method = "fork"

        mp.start_processes(_distributed_worker,
                           nprocs=num_gpu,
                           arg=(main, world_size, num_gpu, args.machine_rank, args.dist_backend, dist_url, (exp, args)),
                           daemon=False,
                           start_method=start_method)
    else:
        data_list = main(exp, args) # 没卡就用CPU
    
    return data_list
