# -*- coding: utf-8 -*-

import os
import cv2
import ast
import copy
import torch
import random
import pprint
import numpy as np
import torch.nn as nn
import torch.distributed as dist

from loguru import logger
from tabulate import tabulate
from yolox.utils import LRScheduler # yolox/utils/lr_scheducler.py

"""Cover yolox/exp/build.py, exps/example/yolox_voc/yolox_voc_s.py"""


class Exp():
    def __init__(self):
        super(Exp, self).__init__()
        
        # ---------------- model config ---------------- #
        self.num_classes = 20 # detect classes number of model
        self.depth = 0.33 # factor of model depth # nano: 0.33 | tiny: 0.330 | s: 0.33 | m: 0.67 | l: 1.0 | x: 1.33
        self.width = 0.50 # factor of model width # nano: 0.25 | tiny: 0.375 | s: 0.50 | m: 0.75 | l: 1.0 | x: 1.25
        self.act = "silu" # activation name. For example, if using "relu", then "silu" will be replaced to "relu"

        # ---------------- dataloader config ---------------- #
        self.data_num_workers = 0 # set worker to 4 for shorter dataloader init time. If your training process cost many memory, reduce this value.
        self.input_size = (512, 512) # (height, width)
        self.multiscale_range = 0 # Actual multiscale ranges: [640 - 5 * 32, 640 + 5 * 32]. To disable multiscale training, set the value to 0.
        # self.random_size = (14, 26) # You can uncomment this line to specify a multiscale range
        self.data_dir = None # dir of dataset images, if data_dir is None, this project will use `datasets` dir
        self.train_ann = "instances_train2017.json" # name of annotation file for training
        self.val_ann = "instances_val2017.json" # name of annotation file for evaluation
        self.test_ann = "instances_test2017.json" # name of annotation file for testing

        # --------------- transform config ----------------- #
        self.mosaic_prob = 1.0 # prob of applying mosaic aug
        self.mixup_prob = 0.0 # prob of applying mixup aug
        self.hsv_prob = 0.0 # prob of applying hsv aug
        self.flip_prob = 0.0 # prob of applying flip aug
        self.degrees = 0.0 # rotation angle range, for example, if set to 2, the true range is (-2, 2)
        self.translate = 0.0 # translate range, for example, if set to 0.1, the true range is (-0.1, 0.1)
        self.mosaic_scale = (0.1, 2)
        self.enable_mixup = False # apply mixup aug or not
        self.mixup_scale = (0.0, 0.0)
        self.shear = 0.0 # shear angle range, for example, if set to 2, the true range is (-2, 2)

        # -------------- training config --------------------- #
        self.warmup_epochs = 1 # epoch number used for warmup
        self.max_epoch = 250 # max training epoch
        self.warmup_lr = 0 # minimum learning rate during warmup
        self.min_lr_ratio = 0.05
        self.basic_lr_per_img = 0.01 / 16.0 # learning rate for one image. During traing, lr will multiply batchsize.
        self.scheduler = "yoloxwarmcos" # name of LRScheduler
        self.no_aug_epochs = 0 # last epoch to close augmention like mosaic
        self.ema = True # apply EMA during training
        
        self.weight_decay = 5e-4 # weight decay of optimizer
        self.momentum = 0.9 # momentum of optimizer
        self.print_interval = 10 # log period in iter, for example, if set to 1, user could see log every iteration.
        self.eval_interval = 10 # eval period in epoch, for example, if set to 1, model will be evaluate after every epoch.
        self.save_history_ckpt = True # save history checkpoint or not, If set to False, yolox will only save latest and best ckpt.
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0] # name of experiment

        # ----------------- testing config ------------------ #
        self.test_size = (512, 512) # output image size during evaluation/test
        self.test_conf = 0.01 # confidence threshold during evalulation/test, boxes whose scores are less than test_conf will be filtered
        self.nmsthre = 0.65 # nms threshold
        
        self.seed = None
        self.traindata  = r'/home/uic/VOCtrainval/'
        self.testdata = r'/home/uic/VOC_test/'
        self.csv_path = "./datapool/VOC0712trainval_ObjectInfo.csv"
        self.output_dir = "./YOLOX_outputs"

    def get_model(self):
        from yolox.models import YOLOX, YOLOPAFPN, YOLOXHead

        def init_yolo(M):
            for m in M.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eps = 1e-3
                    m.momentum = 0.03

        if getattr(self, "model", None) is None:
            in_channels = [256, 512, 1024]
            backbone = YOLOPAFPN(self.depth, self.width, in_channels=in_channels, act=self.act)
            head = YOLOXHead(self.num_classes, self.width, in_channels=in_channels, act=self.act)
            self.model = YOLOX(backbone, head)

        self.model.apply(init_yolo)
        self.model.head.initialize_biases(1e-2)
        return self.model

    def get_data_loader(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import TrainTransform
        from yolox.data.dataloading_rewrite import (VOCDetection,
                                                    YoloBatchSampler,
                                                    DataLoader,
                                                    InfiniteSampler,
                                                    MosaicDetection,
                                                    worker_init_reset_seed)
        
        from yolox.utils import (wait_for_the_master, get_local_rank)
        
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = VOCDetection(data_dir=os.path.join(self.traindata, "VOCdevkit"),
                                   image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                                   img_size=self.input_size,
                                   preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                                   cache=cache_img)

        dataset = MosaicDetection(dataset,
                                  mosaic=not no_aug,
                                  img_size=self.input_size,
                                  preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                                  degrees=self.degrees,
                                  translate=self.translate,
                                  mosaic_scale=self.mosaic_scale,
                                  mixup_scale=self.mixup_scale,
                                  shear=self.shear,
                                  enable_mixup=self.enable_mixup,
                                  mosaic_prob=self.mosaic_prob,
                                  mixup_prob=self.mixup_prob)

        self.dataset = dataset # 经过MosaicDetection就是给dataset实例赋予了更多的函数
        
        if is_distributed: batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0) # 由于可能同时训练voc2007和voc2012因此数据集可能两个

        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug)

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        
        train_loader.__initialized = True
        
        return train_loader

    def get_data_loader_stitcher(self, batch_size, is_distributed, no_aug=False, cache_img=False):
        from yolox.data import TrainTransform
        from yolox.data.dataloading_rewrite import (VOCDetection, YoloBatchSampler, DataLoader_Stitcher, InfiniteSampler, MosaicDetection_Stitcher, worker_init_reset_seed)
        from yolox.utils import (wait_for_the_master, get_local_rank)
        
        local_rank = get_local_rank()

        with wait_for_the_master(local_rank):
            dataset = VOCDetection(data_dir=os.path.join(self.traindata, "VOCdevkit"),
                                   image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                                   img_size=self.input_size,
                                   preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                                   cache=cache_img)
        
        dataset = MosaicDetection_Stitcher(dataset, img_size=self.input_size, preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob))

        self.dataset = dataset # 经过MosaicDetection就是给dataset实例赋予了更多的函数
        
        if is_distributed: batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0) # 由于可能同时训练voc2007和voc2012因此数据集可能两个

        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug)
        
        collator = BatchCollatorSynthesize(self.input_size)

        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed

        dataloader_kwargs["collate_fn"] = collator

        train_loader = DataLoader_Stitcher(self.dataset, **dataloader_kwargs)
        
        train_loader.__initialized = True

        return train_loader

    def get_data_loader_datapool(self, batch_size, is_distributed, csv_path, no_aug=False, cache_img=False):
        from yolox.data import TrainTransform
        from yolox.data.dataloading_rewrite import (VOCDetection, YoloBatchSampler, DataLoader, InfiniteSampler, MosaicDetection_DataPool, worker_init_reset_seed)
        from yolox.utils import (wait_for_the_master, get_local_rank)
        from datapool.datapool_analysis_control import analysis_main # DataPool Analysis Main Function
        from datapool.datapool_syncontrol_v4 import synthesis_preproc # DataPool Analysis Secondary Function
        
        local_rank = get_local_rank()
        
        with wait_for_the_master(local_rank): # 还是输入原始数据集进行dataset的创建
            dataset = VOCDetection(data_dir=os.path.join(self.traindata, "VOCdevkit"),
                                   image_sets=[('2007', 'trainval'), ('2012', 'trainval')],
                                   img_size=self.input_size,
                                   preproc=TrainTransform(max_labels=50, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                                   cache=cache_img)
        
        # 分析整个数据集得到csv文件, 保存到datapool文件夹下
        logger.info("Take a minute to analysis the dataset...")
        if os.path.exists(csv_path) == False:
            analysis_main(dataset._imgpath, dataset._annopath, dataset.ids, csv_path)
            synthesis_preproc(csv_path) # 二次分析csv文件, 节省合成中的过滤合适图块步骤
            
        
        # 使用MosaicDetection_DataPool从csv文件中挑选图块进行拼接
        dataset = MosaicDetection_DataPool(dataset, batch_size, csv_path, img_size=self.input_size, 
                                           preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob))
        
        """dataset = MosaicDetection(dataset,
                                  mosaic=not no_aug,
                                  img_size=self.input_size,
                                  preproc=TrainTransform(max_labels=120, flip_prob=self.flip_prob, hsv_prob=self.hsv_prob),
                                  degrees=self.degrees,
                                  translate=self.translate,
                                  mosaic_scale=self.mosaic_scale,
                                  mixup_scale=self.mixup_scale,
                                  shear=self.shear,
                                  enable_mixup=self.enable_mixup,
                                  mosaic_prob=0.0,
                                  mixup_prob=self.mixup_prob)"""

        self.dataset = dataset # 经过MosaicDetection就是给dataset实例赋予了更多的函数
        
        if is_distributed: batch_size = batch_size // dist.get_world_size()

        sampler = InfiniteSampler(len(self.dataset), seed=self.seed if self.seed else 0) # 由于可能同时训练voc2007和voc2012因此数据集可能两个
        batch_sampler = YoloBatchSampler(sampler=sampler, batch_size=batch_size, drop_last=False, mosaic=not no_aug)
        # collator = BatchCollatorSynthesize(self.input_size)
        dataloader_kwargs = {"num_workers": self.data_num_workers, "pin_memory": True}
        dataloader_kwargs["batch_sampler"] = batch_sampler

        # Make sure each process has different random seed, especially for 'fork' method
        dataloader_kwargs["worker_init_fn"] = worker_init_reset_seed
        # dataloader_kwargs["collate_fn"] = collator
        train_loader = DataLoader(self.dataset, **dataloader_kwargs)
        train_loader.__initialized = True

        return train_loader

    def get_evaluator(self, batch_size, is_distributed, testdev=False, legacy=False):
        from yolox.evaluators import VOCEvaluator
        from yolox.data import ValTransform
        from yolox.data.dataloading_rewrite import VOCDetection

        valdataset = VOCDetection(data_dir=os.path.join(self.testdata, "VOCdevkit"),
                                  image_sets=[('2007', 'test')],
                                  img_size=self.test_size,
                                  preproc=ValTransform(legacy=legacy))

        if is_distributed:
            batch_size = batch_size // dist.get_world_size()
            sampler = torch.utils.data.distributed.DistributedSampler(valdataset, shuffle=False)
        else:
            sampler = torch.utils.data.SequentialSampler(valdataset)

        dataloader_kwargs = {"num_workers": self.data_num_workers,
                             "pin_memory": True,
                             "sampler": sampler}
        
        dataloader_kwargs["batch_size"] = batch_size
        
        val_loader = torch.utils.data.DataLoader(valdataset, **dataloader_kwargs)
        
        evaluator = VOCEvaluator(dataloader=val_loader,
                                 img_size=self.test_size,
                                 confthre=self.test_conf,
                                 nmsthre=self.nmsthre,
                                 num_classes=self.num_classes)
        
        return evaluator
    
    def random_resize(self, data_loader, epoch, rank, is_distributed):
        tensor = torch.LongTensor(2).cuda()

        if rank == 0:
            size_factor = self.input_size[1] * 1.0 / self.input_size[0]
            if not hasattr(self, 'random_size'):
                min_size = int(self.input_size[0] / 32) - self.multiscale_range
                max_size = int(self.input_size[0] / 32) + self.multiscale_range
                self.random_size = (min_size, max_size)
            size = random.randint(*self.random_size)
            size = (int(32 * size), 32 * int(size * size_factor))
            tensor[0] = size[0]
            tensor[1] = size[1]

        if is_distributed:
            dist.barrier()
            dist.broadcast(tensor, 0)

        input_size = (tensor[0].item(), tensor[1].item())
        return input_size

    def preprocess(self, inputs, targets, tsize):
        scale_y = tsize[0] / self.input_size[0]
        scale_x = tsize[1] / self.input_size[1]
        if scale_x != 1 or scale_y != 1:
            inputs = nn.functional.interpolate(inputs, size=tsize, mode="bilinear", align_corners=False)
            targets[..., 1::2] = targets[..., 1::2] * scale_x
            targets[..., 2::2] = targets[..., 2::2] * scale_y
        return inputs, targets

    def get_optimizer(self, batch_size):
        if "optimizer" not in self.__dict__:
            if self.warmup_epochs > 0:
                lr = self.warmup_lr
            else:
                lr = self.basic_lr_per_img * batch_size

            pg0, pg1, pg2 = [], [], [] # optimizer parameter groups

            for k, v in self.model.named_modules():
                if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):
                    pg2.append(v.bias) # biases
                if isinstance(v, nn.BatchNorm2d) or "bn" in k:
                    pg0.append(v.weight) # no decay
                elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):
                    pg1.append(v.weight) # apply decay

            optimizer = torch.optim.SGD(pg0, lr=lr, momentum=self.momentum, nesterov=True)
            optimizer.add_param_group({"params": pg1, "weight_decay": self.weight_decay}) # add pg1 with weight_decay
            optimizer.add_param_group({"params": pg2})
            self.optimizer = optimizer

        return self.optimizer

    def get_lr_scheduler(self, lr, iters_per_epoch):

        scheduler = LRScheduler(self.scheduler, lr, iters_per_epoch, self.max_epoch,
                                warmup_epochs=self.warmup_epochs, warmup_lr_start=self.warmup_lr,
                                no_aug_epochs=self.no_aug_epochs, min_lr_ratio=self.min_lr_ratio)
        
        return scheduler

    def __repr__(self):
        table_header = ["keys", "values"]
        exp_table = [(str(k), pprint.pformat(v)) for k, v in vars(self).items() if not k.startswith("_")]
        return tabulate(exp_table, headers=table_header, tablefmt="fancy_grid")

    def merge(self, cfg_list):
        assert len(cfg_list) % 2 == 0
        for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
            if hasattr(self, k): # only update value with same key
                src_value = getattr(self, k)
                src_type = type(src_value)
                if src_value is not None and src_type != type(v):
                    try:
                        v = src_type(v)
                    except Exception:
                        v = ast.literal_eval(v)
                setattr(self, k, v)


class BatchCollatorSynthesize(object):
    def __init__(self, input_size):
        self.size_divisible = 0
        self.input_size = input_size

    def __call__(self, batch): # batch为list
        transposed_batch = list(zip(*batch)) # 经过测试transposed_batch为list
        images, targets = to_image_list_synthesize_4(transposed_batch, self.input_size, self.size_divisible)
        
        return images, targets, 0, 0


def to_image_list_synthesize_4(transposed_info, input_size, size_divisible=0):
    tensors = transposed_info[0] # batch经过list(zip(*x))处理后的transposed_info是1个列表,其中有1个元组
    if isinstance(tensors, (tuple, list)): # 判断tensors是否属于tuple或list tensors[i].shape=(3, 640, 640)
        targets = transposed_info[1] # targets[i].shape=(120, 5)
        img_ids = transposed_info[2] # ((321, 500), (333, 500), (320, 499), (220, 331), (334, 499), (220, 331), (321, 500), (333, 500))

        # synthesize data:
        assert len(tensors) % 4 == 0, 'len(tensor) % 4 != 0, could not be synthesized ! uneven'
        max_size = tuple(max(s) for s in zip(*[img.shape for img in tensors])) # 取得最大图片的尺寸,(3, 320, 498) c h w
        
        # TODO Ideally, just remove this and let me model handle arbitrary input sizs
        if size_divisible > 0:
            import math
            
            # 将最大宽高的图片尺寸缩放为步长的倍数
            stride = size_divisible
            max_size = list(max_size)
            max_size[1] = int(math.ceil(max_size[1] / stride) * stride) # math.ceil返回输入值的上整数
            max_size[2] = int(math.ceil(max_size[2] / stride) * stride) # math.ceil返回输入值的上整数
            max_size = tuple(max_size)

        batch_shape = (len(tensors)//4,) + max_size # 将两个元组拼接成1个元组
        syn_batched_imgs = torch.from_numpy(tensors[0]).new(*batch_shape).zero_() # syn_batched_imgs.shape = [1, 3, 320, 512]
        # 创造1个batch_shape类型的空张量,并且每处都赋予0值,tensors[0].new()创建1个无值的张量,啥张量后跟new都行
        # 但new()需要输入参数,并且输入的参数不能是列表,因此需要用*batch_shape方式来去除列表
        
        syn_imgs = []
        syn_targets = []
        with torch.no_grad():
            for idx, pad_img in enumerate(syn_batched_imgs): # idx = 0, pad_img.shape = [3, 320, 512], 因为len(tensors)//4==1
                # currently suppose first w then h
                new_h, new_w = max_size[1]//2, max_size[2]//2 # 缩放后尺寸是最大宽高的一半

                # NOTE: interpolate api require first h then w ! interpolate()作用为缩放图片, [c, h, w]
                mode = 'nearest' # squeeze(0)去除第1维
                topLeftImg = torch.nn.functional.interpolate(torch.from_numpy(tensors[idx*4]).unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                topRightImg = torch.nn.functional.interpolate(torch.from_numpy(tensors[idx*4+1]).unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                bottomLeftImg = torch.nn.functional.interpolate(torch.from_numpy(tensors[idx*4+2]).unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                bottomRightImg = torch.nn.functional.interpolate(torch.from_numpy(tensors[idx*4+3]).unsqueeze(0),size=(new_h, new_w),mode=mode).squeeze(0)
                """
                print("pad_img: ", pad_img.shape)
                print("topLeftImg: ", tensors[idx*4].shape, topLeftImg.shape)
                print("topRightImg: ", tensors[idx*4+1].shape, topRightImg.shape)
                print("bottomLeftImg: ", tensors[idx*4+2].shape, bottomLeftImg.shape)
                print("bottomRightImg: ", tensors[idx*4+3].shape, bottomRightImg.shape)
                """
                c = topLeftImg.shape[0] # 取得缩放后图片的通道数
                assert c == topRightImg.shape[0] and c == bottomLeftImg.shape[0] and c == bottomRightImg.shape[0] # 确定缩放后子图片通道数相等
                
                # 当pad_img的宽高不是new_w×2&new_h×2时拼贴会报错
                if topRightImg.shape[1] * 2 != pad_img.shape[1] or topRightImg.shape[2] * 2 != pad_img.shape[2]:
                    pad_img = torch.nn.functional.interpolate(pad_img.unsqueeze(0), size=(new_h * 2, new_w * 2), mode=mode).squeeze(0)
                
                # 将四张缩放后的子图片拼接成一张图片,画个图就很清晰了
                pad_img[:c, :topLeftImg.shape[1], :topLeftImg.shape[2]].copy_(topLeftImg)
                pad_img[:c, :topRightImg.shape[1], topLeftImg.shape[2]:].copy_(topRightImg)
                pad_img[:c, topLeftImg.shape[1]:, :bottomLeftImg.shape[2]].copy_(bottomLeftImg)
                pad_img[:c, topRightImg.shape[1]:, topLeftImg.shape[2]:].copy_(bottomRightImg)
                
                # cv2.imwrite("D:/AICV-YoloXReGPU/abc.jpg", np.transpose(pad_img.numpy(), (1, 2, 0)))
                
                # resize each of four sub-imgs into (new_h, new_w) scale
                # resize api require first w then h ! (120, 5) 120个[cls, x, y, w, h]
                topLeftBL = resize(torch.from_numpy(targets[idx*4]), (tensors[idx*4].shape[2], tensors[idx*4].shape[1]), (new_w, new_h))
                topRightBL = resize(torch.from_numpy(targets[idx*4+1]), (tensors[idx*4+1].shape[2], tensors[idx*4+1].shape[1]), (new_w, new_h))
                bottomLeftBL = resize(torch.from_numpy(targets[idx*4+2]), (tensors[idx*4+2].shape[2], tensors[idx*4+2].shape[1]), (new_w, new_h))
                bottomRightBL = resize(torch.from_numpy(targets[idx*4+3]), (tensors[idx*4+3].shape[2], tensors[idx*4+3].shape[1]), (new_w, new_h))
                
                # 计算四张图片上得到新目标所需的偏移值
                offsets = [torch.Tensor([0.0,0.0,0.0,0.0]), torch.Tensor([new_w,0.0,new_w,0.0]), torch.Tensor([0.0,new_h,0.0,new_h]), torch.Tensor([new_w,new_h,new_w,new_h])]
                
                # append offsets to box coordinates except for topLeftBL 调整GT框坐标到新位置
                topLeftBL = compute_tensor(topLeftBL, offsets[0])
                topRightBL = compute_tensor(topRightBL, offsets[1])
                bottomLeftBL = compute_tensor(bottomLeftBL, offsets[2])
                bottomRightBL = compute_tensor(bottomRightBL, offsets[3])
                
                # 从xyxy还原成xywh格式
                topLeftBL = xyxy_to_xywh(topLeftBL)
                topRightBL = xyxy_to_xywh(topRightBL)
                bottomLeftBL = xyxy_to_xywh(bottomLeftBL)
                bottomRightBL = xyxy_to_xywh(bottomRightBL)
                
                # 填充图片至input_size, default=[640,640] [height, width]
                if pad_img.shape[1] < input_size[0]: # input_size=[height, width], pad_img=[c, h, w]
                    dh = input_size[0] - pad_img.shape[1]
                    dh /= 2
                    pad_top, pad_bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
                else:
                    pad_top, pad_bottom = 0, 0
                
                if pad_img.shape[2] < input_size[1]: # input_size=[height, width], pad_img=[c, h, w]
                    dw = input_size[1] - pad_img.shape[2]
                    dw /= 2
                    pad_left, pad_right = int(round(dw - 0.1)), int(round(dw + 0.1))
                else:
                    pad_left, pad_right = 0, 0
                
                pad_img = cv2.copyMakeBorder(np.transpose(pad_img.numpy(), (1, 2, 0)), pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=(114, 114, 114))
                
                # cv2.imwrite("D:/AICV-YoloXReDST-Orig/after_pad_img.jpg", pad_img)
                
                pad_img = torch.from_numpy(np.transpose(pad_img, (2, 0, 1)))
                
                # 保存图片到列表中,最后拼接成批量
                syn_imgs.append(pad_img.unsqueeze(0))
                
                # 根据填充过程相应地移动边框坐标
                topLeftBL = add_tensor(topLeftBL, pad_left, pad_top)
                topRightBL = add_tensor(topRightBL, pad_left, pad_top)
                bottomLeftBL = add_tensor(bottomLeftBL, pad_left, pad_top)
                bottomRightBL = add_tensor(bottomRightBL, pad_left, pad_top)
                
                """
                topLeft = xywh_to_xyxy(copy.deepcopy(topLeftBL))
                topRight = xywh_to_xyxy(copy.deepcopy(topRightBL))
                bottomLeft = xywh_to_xyxy(copy.deepcopy(bottomLeftBL))
                bottomRight = xywh_to_xyxy(copy.deepcopy(bottomRightBL))
                
                # 可视化拼贴图片的标签是否与目标匹配
                temp_img = np.transpose(copy.deepcopy(pad_img).numpy(), (1, 2, 0)).copy()
                # cv2.imwrite('/mnt/yoloxredstorig/synthesis/syn_img_' + str(random.randint(0, 100000)) + '.jpg', temp_img)
                label_tensor = torch.cat((topLeft,  topRight, bottomLeft, bottomRight), 0)
                _COLORS = np.array([0.000, 0.447, 0.741]).astype(np.float32).reshape(-1, 3)
                for i in range(len(label_tensor)):
                    box = label_tensor[i]
                    x0 = int(box[1])
                    y0 = int(box[2])
                    x1 = int(box[3])
                    y1 = int(box[4])
                    color = (_COLORS[0] * 255).astype(np.uint8).tolist()
                    cv2.rectangle(temp_img, (x0, y0), (x1, y1), color, 2)
                cv2.imwrite('/mnt/yoloxredstorig/synthesis/syn_img_' + str(random.randint(0, 100000)) + '.jpg', temp_img) # cv2.imwrite reqire [h, w, c]
                # print("already done!")
                
                topLeft = xyxy_to_xywh(copy.deepcopy(topLeftBL))
                topRight = xyxy_to_xywh(copy.deepcopy(topRightBL))
                bottomLeft = xyxy_to_xywh(copy.deepcopy(bottomLeftBL))
                bottomRight = xyxy_to_xywh(copy.deepcopy(bottomRightBL))
                """
                
                # 添加0值行变为shape=(120, 5)
                syn_bbox = torch.cat((topLeftBL, topRightBL, bottomLeftBL, bottomRightBL), dim=0)
                zero = torch.tensor([[0., 0., 0., 0., 0.]])
                for i in range(120 - syn_bbox.shape[0]):
                    syn_bbox = torch.cat((syn_bbox, zero),dim=0)
                del zero
                syn_targets.append(syn_bbox.unsqueeze(0))
        
        # 检查ID数量是否也为4的倍数
        assert len(img_ids)%4 == 0
        
        # 拼接合成目标与合成标签为batch张量
        syn_imgs = torch.cat(syn_imgs, dim=0)
        syn_targets = torch.cat(syn_targets, dim=0)
        
        return syn_imgs, syn_targets
    else:
        raise TypeError("Unsupported type for to_image_list: {}".format(type(tensors)))


def resize(targets, sizeori, sizenew): # 输入xywh返回xyxy
    # 去掉标签张量中的0值行
    temp = []
    for i in range(targets.shape[0]):
        if targets[i][3]!=0 and targets[i][4]!=0:
            temp.append(targets[i])
    targets = torch.stack(temp, dim=0)
    del temp

    # 取得新宽高与旧宽高的比例元组
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(sizenew, sizeori))

    # 当宽高新旧比例不等时需要分别操作
    ratio_width, ratio_height = ratios
    for i in range(targets.shape[0]):
        xmin = targets[i][1] - targets[i][3]/2
        ymin = targets[i][2] - targets[i][4]/2
        xmax = targets[i][1] + targets[i][3]/2
        ymax = targets[i][2] + targets[i][4]/2
        scaled_xmin = xmin * ratio_width
        scaled_xmax = xmax * ratio_width
        scaled_ymin = ymin * ratio_height
        scaled_ymax = ymax * ratio_height
        targets[i][1] = scaled_xmin
        targets[i][2] = scaled_ymin
        targets[i][3] = scaled_xmax
        targets[i][4] = scaled_ymax
    
    return targets


def compute_tensor(tensor1, tensor2):
    for i in range(tensor1.shape[0]):
        tensor1[i][1:] = tensor1[i][1:] + tensor2
    
    return tensor1


def add_tensor(tensor1, pad_left, pad_top): # 专门用于为xcen&ycen添加填充量
    for i in range(tensor1.shape[0]):
        tensor1[i][1] = tensor1[i][1] + pad_left
        tensor1[i][2] = tensor1[i][2] + pad_top

    return tensor1


def xyxy_to_xywh(tensor): # 输入xyxy返回xywh
    for i in range(tensor.shape[0]):
        scaled_xcen = (tensor[i][3] + tensor[i][1]) / 2
        scaled_ycen = (tensor[i][4] + tensor[i][2]) / 2
        scaled_w = tensor[i][3] - tensor[i][1]
        scaled_h = tensor[i][4] - tensor[i][2]
        tensor[i][1] = scaled_xcen
        tensor[i][2] = scaled_ycen
        tensor[i][3] = scaled_w
        tensor[i][4] = scaled_h
    
    return tensor


def xywh_to_xyxy(tensor): # [cls, x1, y1, x2, y2]
    for i in range(tensor.shape[0]):
        x1 = tensor[i][1] - tensor[i][3] / 2
        y1 = tensor[i][2] - tensor[i][4] / 2
        x2 = tensor[i][1] + tensor[i][3] / 2
        y2 = tensor[i][2] + tensor[i][4] / 2
        tensor[i][1] = x1
        tensor[i][2] = y1
        tensor[i][3] = x2
        tensor[i][4] = y2
        
    return tensor
