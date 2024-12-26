# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.
import os
import cv2
import time
import math
import copy
import torch
import pickle
import random
import weakref
import logging
import statistics
import numpy as np
import concurrent.futures
import matplotlib.pyplot as plt

from typing import List, Mapping, Optional
from torch.nn.parallel import DataParallel, DistributedDataParallel

import detectron2.utils.comm as comm

from detectron2.utils.logger import _log_api_usage
from detectron2.utils.events import EventStorage, get_event_storage

import warnings
warnings.filterwarnings("ignore")

__all__ = ["HookBase", "TrainerBase", "SimpleTrainer", "AMPTrainer"]


class HookBase:
    """Base class for hooks that can be registered with :class:`TrainerBase`.

    Each hook can implement 4 methods. The way they are called is demonstrated
    in the following snippet:
    ::
        hook.before_train()
        for iter in range(start_iter, max_iter):
            hook.before_step()
            trainer.run_step()
            hook.after_step()
        iter += 1
        hook.after_train()

    Notes:
        1. In the hook method, users can access ``self.trainer`` to access more
           properties about the context (e.g., model, current iteration, or config
           if using :class:`DefaultTrainer`).

        2. A hook that does something in :meth:`before_step` can often be
           implemented equivalently in :meth:`after_step`.
           If the hook takes non-trivial time, it is strongly recommended to
           implement the hook in :meth:`after_step` instead of :meth:`before_step`.
           The convention is that :meth:`before_step` should only take negligible time.

           Following this convention will allow hooks that do care about the difference
           between :meth:`before_step` and :meth:`after_step` (e.g., timer) to
           function properly."""

    trainer: "TrainerBase" = None
    
    """A weak reference to the trainer object. Set by the trainer when the hook is registered."""

    def before_train(self):
        """Called before the first iteration."""
        pass

    def after_train(self):
        """Called after the last iteration."""
        pass

    def before_step(self):
        """Called before each iteration."""
        pass

    def after_backward(self):
        """Called after the backward pass of each iteration."""
        pass

    def after_step(self):
        """Called after each iteration."""
        pass

    def state_dict(self):
        """Hooks are stateless by default, but can be made checkpointable by implementing `state_dict` and `load_state_dict`."""
        return {}


class TrainerBase:
    """Base class for iterative trainer with hooks.

    The only assumption we made here is: the training runs in a loop.
    A subclass can implement what the loop is.
    We made no assumptions about the existence of dataloader, optimizer, model, etc.

    Attributes:
        iter(int): the current iteration.

        start_iter(int): The iteration to start with.
            By convention the minimum possible value is 0.

        max_iter(int): The iteration to end training.

        storage(EventStorage): An EventStorage that's opened during the course of training."""

    def __init__(self) -> None:
        self._hooks: List[HookBase] = []
        self.iter: int = 0
        self.start_iter: int = 0
        self.max_iter: int
        self.storage: EventStorage
        _log_api_usage("trainer." + self.__class__.__name__)

    def register_hooks(self, hooks: List[Optional[HookBase]]) -> None:
        """Register hooks to the trainer. The hooks are executed in the order they are registered.
        Args: hooks (list[Optional[HookBase]]): list of hooks"""
        
        hooks = [h for h in hooks if h is not None]
        for h in hooks:
            assert isinstance(h, HookBase)
            # To avoid circular reference, hooks and trainer cannot own each other.
            # This normally does not matter, but will cause memory leak if the
            # involved objects contain __del__:
            # See http://engineering.hearsaysocial.com/2013/06/16/circular-references-in-python/
            h.trainer = weakref.proxy(self)
        self._hooks.extend(hooks)

    def train(self, start_iter: int, max_iter: int):
        """Args: start_iter, max_iter (int): See docs above"""
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step(self.iter, max_iter)
                    self.after_step()
                # self.iter == max_iter can be used by `after_train` to tell whether the training successfully finished or failed due to exceptions.
                self.iter += 1
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def before_train(self):
        for h in self._hooks:
            h.before_train()

    def after_train(self):
        self.storage.iter = self.iter
        for h in self._hooks:
            h.after_train()

    def before_step(self):
        # Maintain the invariant that storage.iter == trainer.iter for the entire execution of each step
        self.storage.iter = self.iter
        for h in self._hooks: h.before_step()

    def after_backward(self):
        for h in self._hooks: h.after_backward()

    def after_step(self):
        for h in self._hooks: h.after_step()

    def run_step(self, iter_num):
        raise NotImplementedError

    def state_dict(self):
        ret = {"iteration": self.iter}
        hooks_state = {}
        for h in self._hooks:
            sd = h.state_dict()
            if sd:
                name = type(h).__qualname__
                if name in hooks_state: continue
                hooks_state[name] = sd
        if hooks_state: ret["hooks"] = hooks_state
        return ret

    def load_state_dict(self, state_dict):
        logger = logging.getLogger(__name__)
        self.iter = state_dict["iteration"]
        for key, value in state_dict.get("hooks", {}).items():
            for h in self._hooks:
                try:
                    name = type(h).__qualname__
                except AttributeError:
                    continue
                if name == key:
                    h.load_state_dict(value)
                    break
            else: logger.warning(f"Cannot find the hook '{key}', its state_dict is ignored.")


class DataPrefetcher:
    def __init__(self, loader):
        self.loader = iter(loader)
        self.preload()

    def preload(self):
        try:
            self.data = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_target = None
            return
    def next(self):
        data = self.data
        self.preload()
        return data


from detectron2.structures import Boxes, Instances
# from datapool.datapool_syncontrol_v4 import synthesis_control # DataPool Synthesis Primary Function

class DataPrefetcher_datapool:
    def __init__(self):
        self.objscale = 1024
        self.objheight = 640
        self.objwidth = 1024
        self.coco_path = 'F:/coco/'
        
        # basic path: path + main file
        dp_path, train_size = "D:/AICV-Detectron2/VOC_0712trainval_ObjectInfo.csv", 1
        
        # assemble main csv file absolute path
        if train_size == 1: 
            self.main_csv = os.path.dirname(dp_path) + '/dp_' + str(self.objscale) + '_full/' + os.path.basename(dp_path)[0:-4] + '_' + str(self.objscale) + '.csv'
        if train_size == 0.5: 
            self.main_csv = os.path.dirname(dp_path) + '/dp_' + str(self.objscale) + '_half/' + os.path.basename(dp_path)[0:-4] + '_' + str(self.objscale) + '.csv'
        if train_size == 0.25: 
            self.main_csv = os.path.dirname(dp_path) + '/dp_' + str(self.objscale) + '_quar/' + os.path.basename(dp_path)[0:-4] + '_' + str(self.objscale) + '.csv'
        
        # DataPool: 分析整个数据集并保存csv文件
        if os.path.exists(self.main_csv) == False:
            print("Take a minute to analysis the dataset...")
            analysis_main_coco(self.coco_path + 'images/train2017/', self.coco_path + 'annotations/instances_train2017.json', self.main_csv)
        
        # 二次分析csv文件, 节省合成中的过滤合适图块步骤
        if os.path.exists(os.path.dirname(self.main_csv) + '/VOC_0712trainval_config_1_' + str(self.objscale) + '_area1.csv') == False:
            print("Take another minute to analysis the csv file...")
            synthesis_preproc(self.main_csv, self.objscale, self.objheight, self.objwidth)
        
        # instanclize synthesis control for time saving
        self.syn_con = synthesis_control(self.main_csv, self.objscale, self.objheight, self.objwidth)
        self.batch_size = 2
        
        # preload once for first time forward propagation
        self.data = 0
        self.preload()

    def preload(self):
        data = []
        # current form: img.shape=[h, w, c]numpy, lab.shape=[N, 5]torch, xyxycls
        # intend form: img.shape=[c, h, w], gt_classes=cls, gt_boxes=[N, 5][xyxy]
        # data is list, len(data)=bs, data[i]=dict, dict.key()=image, instances
        try:
            img_list, lab_list = self.syn_con.synthesis_process(self.batch_size)
        except Exception as e:
            return 
        
        for i in range(len(img_list)):
            img_list[i] = torch.from_numpy(cv2.cvtColor(img_list[i], cv2.COLOR_BGR2RGB))
            
            # img.shape [h, w, c]->[c, h, w]
            img_list[i] = img_list[i].permute(2, 0, 1)
            
            # divide annots into gt_classes & gt_boxes
            gt_classes = lab_list[i][..., 4]
            gt_boxes = lab_list[i][...,0:4]
            
            # build up Boxes & instances with boxes & classes
            # have to be this way when building instances
            img_size = (int(img_list[i].shape[1]), int(img_list[i].shape[2]))
            instances = Instances(img_size)
            instances.gt_boxes = Boxes(gt_boxes)
            instances.gt_classes = gt_classes
            
            # append one img's dict into data
            data.append({'image':img_list[i], 'instances': instances})
        self.data = data

    def next(self):
        data = self.data
        self.preload()
        return data

# inps: [c, h, w], targets: [N, line]
def visualize_inputs(inps, targets):
    _COLORS = np.array([0.000, 0.447, 0.741, # blue
                        0.850, 0.325, 0.098, # orange
                        0.929, 0.694, 0.125, # yellow
                        0.494, 0.184, 0.556, # purple
                        0.466, 0.674, 0.188, # light green
                        0.301, 0.745, 0.933, # light blue
                        0.635, 0.078, 0.184, # deep red
                        # 0.300, 0.300, 0.300, # grey
                        # 0.600, 0.600, 0.600, # light grey
                        1.000, 0.000, 0.000, # red
                        1.000, 0.500, 0.000, # light orange
                        0.749, 0.749, 0.000, # green
                        0.000, 1.000, 0.000, # other green
                        0.000, 0.000, 1.000, # deep blue
                        0.667, 0.000, 1.000, # other purple
                        # 0.333, 0.333, 0.000, # grey yellow
                        0.333, 0.667, 0.000, # another green
                        0.333, 1.000, 0.000, # bright green
                        ]).astype(np.float32).reshape(-1, 3)
    
    # 处理图片
    temp_img = np.ascontiguousarray(copy.deepcopy(inps).permute(1, 2, 0).numpy())

    for box in  targets:
        x0, y0, x1, y1 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = (_COLORS[random.randint(0, 14)] * 255).astype(np.uint8).tolist()
        cv2.rectangle(temp_img, (x0, y0), (x1, y1), color, 2) # cv2 takes numpy
    
    # temp_img = cv2.cvtColor(temp_img, cv2.COLOR_BGR2RGB) # if readin by PIL.Image, then comment out this line
    cv2.imwrite('./tbtest/' + str(random.randint(1, 500)) + '_test.jpg', temp_img) # cv2.imwrite reqire [h, w, c]


class SimpleTrainer(TrainerBase):
    """A simple trainer for the most common type of task:
    single-cost single-optimizer single-data-source iterative optimization, optionally using data-parallelism.
    It assumes that every step, you:

    1. Compute the loss with a data from the data_loader.
    2. Compute the gradients with the above loss.
    3. Update the model with the optimizer.

    All other tasks during training (checkpointing, logging, evaluation, LR schedule)
    are maintained by hooks, which can be registered by :meth:`TrainerBase.register_hooks`.

    If you want to do anything fancier than this, either subclass TrainerBase and implement your own `run_step`,
    or write your own training loop."""
    
    def __init__(self, model, data_loader, optimizer, gather_metric_period=1, zero_grad_before_forward=False, async_write_metrics=False):
        """Args:
            model: a torch Module. Takes a data from data_loader and returns a dict of losses.
            data_loader: an iterable. Contains data to be used to call model.
            optimizer: a torch optimizer.
            gather_metric_period: an int. Every gather_metric_period iterations the metrics are gathered from all the ranks to rank 0 and logged.
            zero_grad_before_forward: whether to zero the gradients before the forward.
            async_write_metrics: bool. If True, then write metrics asynchronously to improve training speed"""
        
        super().__init__()

        """We set the model to training mode in the trainer.
        However it's valid to train a model that's in eval mode.
        If you want your model (or a submodule of it) to behave
        like evaluation during training, you can overwrite its train() method."""
        
        model.train()

        self.model = model
        self.data_loader = data_loader[0] # to access the data loader iterator, call `self._data_loader_iter`
        self.data_loader_st = data_loader[1]
        self.data_loader_dp = data_loader[2]
        if self.data_loader != None:
            self.prefetcher_regular = DataPrefetcher(self.data_loader)
        if self.data_loader_st != None:
            self.prefetcher_stitcher = DataPrefetcher(self.data_loader_st)
        if self.data_loader_dp != None:
            self.prefetcher_datapool = DataPrefetcher_datapool()
        self._data_loader_iter_obj = None
        self.optimizer = optimizer
        self.gather_metric_period = gather_metric_period
        self.zero_grad_before_forward = zero_grad_before_forward
        self.async_write_metrics = async_write_metrics
        
        # addition code: initialize object scale ratio
        # self.ratio_scale, self.batch_flag = [1.0, 1.0, 1.0], 0
        
        # addition code: initialize margin focal params
        self.obj_mgn_list, self.obj_mgn_max = [[], [], []], [-3, -3, -3]
        self.cls_mgn_list, self.cls_mgn_max = [[], [], []], [-3, -3, -3]
        
        # create a thread pool that can execute non critical logic in run_step asynchronically
        # use only 1 worker so tasks will be executred in order of submitting.
        self.concurrent_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    def run_step(self, iter_num, max_iter):
        """Implement the standard training logic described above."""
        
        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        
        # start timing data fetching
        start = time.perf_counter()
        
        """If you want to do something with the data, you can wrap the dataloader."""
        
        # data = self.dynamic_controller_stitcher()
        # data = self.dynamic_controller_datapool()
        # data = self.hybrid_controller(iter_num)
        # data = self.global_controller(iter_num)
    
        # fetch next batch data
        # data = next(self._data_loader_iter)
        data = self.prefetcher_regular.next()
        
        # visualize images from this batch
        # for i in range(len(data)):
        #     visualize_inputs_dp(data[i]['image'].unsqueeze(0), data[i]['instances'].gt_boxes.tensor.unsqueeze(0).numpy())
        
        # end timing data fetching
        data_time = time.perf_counter() - start
        
        # clean up gradient
        if self.zero_grad_before_forward:
            """If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method."""
            self.optimizer.zero_grad()

        """If you want to do something with the losses, you can wrap the model."""
        
        # print("Form of data: ", len(data), data[0]['image'].shape, type(data[0]['image']), data[0].keys(), type(data[0]['instances']))
        # data is list, len(data)=bs, data[0].keys()=dict_keys(['file_name', 'height', 'width', 'image_id', 'image', 'instances'])
        
        # print("Form of instances: ", len(data[0]['instances']), data[0]['instances']._image_size[0], data[0]['instances']._image_size[1])
        # num_instances, image_height, image_width, fields, # instances return a dict, including gt_boxes and gt_classes as keys
        
        # print("Form of GT Boxes & Classes: ", data[0]['instances']._fields['gt_boxes'].tensor, data[0]['instances']._fields['gt_classes']) 
        # Box tensor [N, 4] xyxy, Classes tensor([55, 55, 55, 55, 55, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 49, 55, 55, 60])
        
        # forward propagation
        loss_dict, self.cls_mgn_list, self.cls_mgn_max = self.model(data, self.obj_mgn_list, self.obj_mgn_max, 
                                                                    self.cls_mgn_list, self.cls_mgn_max, iter_num, max_iter)
        
        # addition code: get object scale info
        self.obj_mgn_list = loss_dict['margin_list']
        self.obj_mgn_max = loss_dict['margin_max']

        # addition code: delete addition info
        # loss_dict.pop('ratio_scale')
        loss_dict.pop('margin_list')
        loss_dict.pop('margin_max')
        
        # sum all the loss items
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())
        
        # clean up gradient
        if not self.zero_grad_before_forward:
            """If you need to accumulate gradients or do something similar, you can
            wrap the optimizer with your custom `zero_grad()` method."""
            self.optimizer.zero_grad()
        
        # back propagation
        losses.backward()

        self.after_backward()

        if self.async_write_metrics: # write metrics asynchronically
            self.concurrent_executor.submit(self._write_metrics, loss_dict, data_time, iter=self.iter)
        else:
            self._write_metrics(loss_dict, data_time)
        
        """If you need gradient clipping/scaling or other processing, you can
        wrap the optimizer with your custom `step()` method. But it is
        suboptimal as explained in https://arxiv.org/abs/2006.15704 Sec 3.2.4"""
        
        # update optimizer
        self.optimizer.step()
        
    @property
    def thr_regulator_cosine(self, max_epoch, epoch):
        return 1 - self.yolox_warm_cos_stitcher(0.1, max_epoch, epoch)
    
    # @property
    # def thr_regulator_step(self, max_epoch, epoch):
    #     return 1 - self.yolox_warm_cos_stitcher(0.1, max_epoch, epoch)
    
    def yolox_warm_cos_stitcher(self, prob, total_epoch, current_epoch):
        prob = 0.5 * prob * (1.0 + math.cos(math.pi * current_epoch / total_epoch))
        return prob

    def dynamic_controller_datapool(self):
        if self.ratio_scale[0] < 0.1:
            if self.batch_flag != 0:
                data = self.prefetcher_regular.next()
                self.batch_flag = 0
            else:
                if random.randint(0, 10) <= 5:
                    data = self.prefetcher_datapool.next()
                    self.batch_flag = 2
                else:
                    data = self.prefetcher_stitcher.next()
                    self.batch_flag = 1
        else:
            data = self.prefetcher_regular.next()
            self.batch_flag = 0
        return data

    def dynamic_controller_stitcher(self):
        if self.ratio_scale[0] < 0.1:
            if self.batch_flag != 0:
                data = self.prefetcher_regular.next()
                self.batch_flag = 0
            else:
                data = self.prefetcher_stitcher.next()
                self.batch_flag = 2
        else:
            data = self.prefetcher_regular.next()
            self.batch_flag = 0
        return data

    def global_controller(self, iter_num):
        if random.random() < self.thr_regulator_cosine(90000 / 1000, math.floor(iter_num / 1000)): # max_epoch, epoch_num
            data = self.prefetcher_regular.next()
        else:
            data = self.prefetcher_stitcher.next()
        return data

    def hybrid_controller(self, iter_num):
        if self.ratio_scale[0] < 0.1:
            if self.batch_flag != 0:
                data = self.prefetcher_regular.next()
                self.batch_flag = 0
            else:
                if random.random() * 100 <= self.yolox_warm_cos_stitcher(0.1, 90, math.floor(iter_num / 1000)) * 1000:
                    data = self.prefetcher_stitcher.next()
                    self.batch_flag = 1
                else:
                    data = self.prefetcher_regular.next()
                    self.batch_flag = 0
        else:
            data = self.prefetcher_regular.next()
            self.batch_flag = 0
        return data

    @property
    def _data_loader_iter(self):
        # only create the data loader iterator when it is used
        if self._data_loader_iter_obj is None:
            self._data_loader_iter_obj = iter(self.data_loader)
        return self._data_loader_iter_obj

    def reset_data_loader(self, data_loader_builder):
        """Delete and replace the current data loader with a new one, which will be created by calling `data_loader_builder` (without argument)."""
        del self.data_loader
        data_loader = data_loader_builder()
        self.data_loader = data_loader
        self._data_loader_iter_obj = None

    def _write_metrics(self, loss_dict: Mapping[str, torch.Tensor], data_time: float, prefix: str = "", iter: Optional[int] = None) -> None:
        logger = logging.getLogger(__name__)

        iter = self.iter if iter is None else iter
        if (iter + 1) % self.gather_metric_period == 0:
            try:
                SimpleTrainer.write_metrics(loss_dict, data_time, iter, prefix)
            except Exception:
                logger.exception("Exception in writing metrics: ")
                raise

    @staticmethod
    def write_metrics(loss_dict: Mapping[str, torch.Tensor], data_time: float, cur_iter: int, prefix: str = "") -> None:
        """Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
            prefix (str): prefix for logging keys"""
        
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        storage = get_event_storage()
        # Keep track of data time per rank
        storage.put_scalar("rank_data_time", data_time, cur_iter=cur_iter)

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time, cur_iter=cur_iter)

            # average the rest metrics
            metrics_dict = {k: np.mean([x[k] for x in all_metrics_dict]) for k in all_metrics_dict[0].keys()}
            total_losses_reduced = sum(metrics_dict.values())
            if not np.isfinite(total_losses_reduced):
                raise FloatingPointError(f"Loss became infinite or NaN at iteration={cur_iter}!\n"
                                         f"loss_dict = {metrics_dict}")
                

            storage.put_scalar("{}total_loss".format(prefix), total_losses_reduced, cur_iter=cur_iter)
            if len(metrics_dict) > 1: storage.put_scalars(cur_iter=cur_iter, **metrics_dict)

    def state_dict(self):
        ret = super().state_dict()
        ret["optimizer"] = self.optimizer.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.optimizer.load_state_dict(state_dict["optimizer"])

    def after_train(self):
        super().after_train()
        self.concurrent_executor.shutdown(wait=True)


class AMPTrainer(SimpleTrainer):
    """Like :class:`SimpleTrainer`, but uses PyTorch's native automatic mixed precisionin the training loop."""
    def __init__(self, model, data_loader, optimizer, gather_metric_period=1, zero_grad_before_forward=False, grad_scaler=None, precision: torch.dtype = torch.float16,
                 log_grad_scaler: bool = False, async_write_metrics=False):
        
        """Args:
            model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward,
                async_write_metrics: same as in :class:`SimpleTrainer`.
            grad_scaler: torch GradScaler to automatically scale gradients.
            precision: torch.dtype as the target precision to cast to in computations"""
        
        unsupported = "AMPTrainer does not support single-process multi-device training!"
        if isinstance(model, DistributedDataParallel):
            assert not (model.device_ids and len(model.device_ids) > 1), unsupported
        assert not isinstance(model, DataParallel), unsupported

        super().__init__(model, data_loader, optimizer, gather_metric_period, zero_grad_before_forward)

        if grad_scaler is None:
            from torch.cuda.amp import GradScaler
            grad_scaler = GradScaler()
        self.grad_scaler = grad_scaler
        self.precision = precision
        self.log_grad_scaler = log_grad_scaler

    def run_step(self):
        """Implement the AMP training logic."""
        
        assert self.model.training, "[AMPTrainer] model was changed to eval mode!"
        assert torch.cuda.is_available(), "[AMPTrainer] CUDA is required for AMP training!"
        from torch.cuda.amp import autocast

        start = time.perf_counter()
        data = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        if self.zero_grad_before_forward:
            self.optimizer.zero_grad()
        with autocast(dtype=self.precision):
            loss_dict = self.model(data)
            if isinstance(loss_dict, torch.Tensor):
                losses = loss_dict
                loss_dict = {"total_loss": loss_dict}
            else:
                losses = sum(loss_dict.values())

        if not self.zero_grad_before_forward:
            self.optimizer.zero_grad()

        self.grad_scaler.scale(losses).backward()

        if self.log_grad_scaler:
            storage = get_event_storage()
            storage.put_scalar("[metric]grad_scaler", self.grad_scaler.get_scale())

        self.after_backward()

        if self.async_write_metrics: # write metrics asynchronically
            self.concurrent_executor.submit(self._write_metrics, loss_dict, data_time, iter=self.iter)
        else:
            self._write_metrics(loss_dict, data_time)

        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

    def state_dict(self):
        ret = super().state_dict()
        ret["grad_scaler"] = self.grad_scaler.state_dict()
        return ret

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        self.grad_scaler.load_state_dict(state_dict["grad_scaler"])
