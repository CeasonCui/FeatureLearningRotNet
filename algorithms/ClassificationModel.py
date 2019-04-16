# -*- coding: UTF-8 -*-
from __future__ import print_function
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import os
import torchnet as tnt
import utils
import PIL
import pickle
from tqdm import tqdm
import time

from . import Algorithm
from pdb import set_trace as breakpoint

    
"""准确率计算函数
    输入output是模型预测的结果，
    尺寸为batch size*num class；target是真实标签，长度为batch size。
    这二者都是Tensor类型，具体而言前者是Float Tensor，后者是Long Tensor。
    batch_size = target.size(0)是读取batch size值。 
    _, pred = output.topk(maxk, 1, True, True)这里调用了PyTorch中Tensor的topk方法，
    第一个输入maxk表示你要计算的是top maxk的结果；
    第二个输入1表示dim，即按行计算（dim=1）；
    第三个输入True完整的是largest=True，表示返回的是top maxk个最大值；
    第四个输入True完整的是sorted=True，表示返回排序的结果，主要是因为后面要基于这个top maxk的结果计算top 1。
    target.view(1, -1).expand_as(pred)先将target的尺寸规范到1*batch size，
    然后将维度扩充为pred相同的维度，也就是maxk*batch size，比如5*batch size，
    然后调用eq方法计算两个Tensor矩阵相同元素情况，得到的correct是同等维度的ByteTensor矩阵，1值表示相等，0值表示不相等。
    correct_k = correct[:k].view(-1).float().sum(0)通过k值来决定是计算top k的准确率，
    sum(0)表示按照dim 0维度计算和，最后都添加到res列表中并返回。
"""
#topk(k, dim=None, largest=True, sorted=True) -> (Tensor, LongTensor)
#返回给定维度上给定输入张量的k个最大元素。
#如果未给出维度，则选择输入的最后一个维度。
#如果maximum为False，则返回k个最小元素。
#返回（values，indices）的命名元组，其中索引是原始输入张量中元素的索引。
#如果为True，则布尔选项将确保返回的k元素本身已排序
def accuracy(output, target, topk=(1,)): #准确率计算函数
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0) #批量大小
    _, pred = output.topk(maxk, 1, True, True) #返回最大元素
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class ClassificationModel(Algorithm):
    def __init__(self, opt):
        Algorithm.__init__(self, opt)

    def allocate_tensors(self):#分配张量
        self.tensors = {}
        self.tensors['dataX'] = torch.FloatTensor()
        self.tensors['labels'] = torch.LongTensor()

    def train_step(self, batch):
        return self.process_batch(batch, do_train=True)

    def evaluation_step(self, batch):
        return self.process_batch(batch, do_train=False)

    def process_batch(self, batch, do_train=True):
        #*************** LOAD BATCH (AND MOVE IT TO GPU) ********
        #加载批量（并将其移至GPU）
        start = time.time()
        self.tensors['dataX'].resize_(batch[0].size()).copy_(batch[0])#为啥这里分batch0 batch1
        self.tensors['labels'].resize_(batch[1].size()).copy_(batch[1])
        dataX = self.tensors['dataX']
        labels = self.tensors['labels']
        batch_load_time = time.time() - start
        #********************************************************

        #********************************************************
        start = time.time()
        if do_train: # zero the gradients #梯度
            self.optimizers['model'].zero_grad() #每次都要置零
        #********************************************************

        #***************** SET TORCH VARIABLES ******************
        dataX_var = torch.autograd.Variable(dataX, volatile=(not do_train))
        labels_var = torch.autograd.Variable(labels, requires_grad=False)
        #********************************************************

        #************ FORWARD THROUGH NET ***********************
        pred_var = self.networks['model'](dataX_var)
        #********************************************************

        #*************** COMPUTE LOSSES *************************
        record = {}
        loss_total = self.criterions['loss'](pred_var, labels_var)
        record['prec1'] = accuracy(pred_var.data, labels, topk=(1,))[0].item() #max(topk)=1,这是个top1
        record['loss'] = loss_total.data.item()
        #********************************************************

        #****** BACKPROPAGATE AND APPLY OPTIMIZATION STEP *******
        if do_train:
            loss_total.backward()
            self.optimizers['model'].step() #update
        #********************************************************
        batch_process_time = time.time() - start
        total_time = batch_process_time + batch_load_time
        record['load_time'] = 100*(batch_load_time/total_time)
        record['process_time'] = 100*(batch_process_time/total_time)

        return record
