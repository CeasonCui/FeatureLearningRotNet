import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from pdb import set_trace as breakpoint

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class AlexNet(nn.Module):
    def __init__(self, opt):
        super(AlexNet, self).__init__()
        num_classes = opt['num_classes']
#class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True)
#n_channels(int) – 输入信号的通道
#out_channels(int) – 卷积产生的通道
#kerner_size(int or tuple) - 卷积核的尺寸
#stride(int or tuple, optional) - 卷积步长
#padding(int or tuple, optional) - 输入的每一条边补充0的层数
#dilation(int or tuple, optional) – 卷积核元素之间的间距
#groups(int, optional) – 从输入通道到输出通道的阻塞连接数
#bias(bool, optional) - 如果bias=True，添加偏置

#class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True)
#num_features： 来自期望输入的特征数，该期望输入的大小为'batch_size x num_features x height x width'
#eps： 为保证数值稳定性（分母不能趋近或取0）,给分母加上的值。默认为1e-5。
#momentum： 动态均值和动态方差所使用的动量。默认为0.1。
#affine： 一个布尔值，当设为true，给该层添加可学习的仿射变换参数。

        conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv2 = nn.Sequential(
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
        )
        pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        conv3 = nn.Sequential(
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
        )
        conv4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        conv5 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        pool5 = nn.MaxPool2d(kernel_size=3, stride=2)
        num_pool5_feats = 6 * 6 * 256
#class torch.nn.Linear(in_features, out_features, bias=True)
#in_features - 每个输入样本的大小
#out_features - 每个输出样本的大小
#bias - 若设置为False，这层不会学习偏置。默认值：True
        fc_block = nn.Sequential(
            Flatten(),
            nn.Linear(num_pool5_feats, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )
        classifier = nn.Sequential(
            nn.Linear(4096, num_classes),
        )

        self._feature_blocks = nn.ModuleList([
    		conv1,
    		pool1,
    		conv2,
    		pool2,
    		conv3,
    		conv4,
    		conv5,
    		pool5,
    		fc_block,
    		classifier,
        ])
        self.all_feat_names = [
    		'conv1',
    		'pool1',
    		'conv2',
    		'pool2',
    		'conv3',
    		'conv4',
    		'conv5',
    		'pool5',
    		'fc_block',
    		'classifier',
        ]
        assert(len(self.all_feat_names) == len(self._feature_blocks))

    def _parse_out_keys_arg(self, out_feat_keys):

    	# By default return the features of the last layer / module.
    	out_feat_keys = [self.all_feat_names[-1],] if out_feat_keys is None else out_feat_keys

        if len(out_feat_keys) == 0:
    		raise ValueError('Empty list of output feature keys.')
        for f, key in enumerate(out_feat_keys):
    		if key not in self.all_feat_names:
    			raise ValueError('Feature with name {0} does not exist. Existing features: {1}.'.format(key, self.all_feat_names))
    		elif key in out_feat_keys[:f]:
    			raise ValueError('Duplicate output feature key: {0}.'.format(key))

    	# Find the highest output feature in `out_feat_keys
    	max_out_feat = max([self.all_feat_names.index(key) for key in out_feat_keys])

    	return out_feat_keys, max_out_feat

    def forward(self, x, out_feat_keys=None):
    	"""Forward an image `x` through the network and return the asked output features.

    	Args:
    	  x: input image.
    	  out_feat_keys: a list/tuple with the feature names of the features
                that the function should return. By default the last feature of
                the network is returned.

    	Return:
            out_feats: If multiple output features were asked then `out_feats`
                is a list with the asked output features placed in the same
                order as in `out_feat_keys`. If a single output feature was
                asked then `out_feats` is that output feature (and not a list).
    	"""
    	out_feat_keys, max_out_feat = self._parse_out_keys_arg(out_feat_keys)
    	out_feats = [None] * len(out_feat_keys)

    	feat = x
    	for f in range(max_out_feat+1):
    		feat = self._feature_blocks[f](feat)
    		key = self.all_feat_names[f]
    		if key in out_feat_keys:
    			out_feats[out_feat_keys.index(key)] = feat

        out_feats = out_feats[0] if len(out_feats)==1 else out_feats
    	return out_feats

    def get_L1filters(self):
        convlayer = self._feature_blocks[0][0]
        batchnorm = self._feature_blocks[0][1]
        filters = convlayer.weight.data
        scalars = (batchnorm.weight.data / torch.sqrt(batchnorm.running_var + 1e-05))
        filters = (filters * scalars.view(-1, 1, 1, 1).expand_as(filters)).cpu().clone()

        return filters

def create_model(opt):
    return AlexNet(opt)

if __name__ == '__main__':
    size = 224
    opt = {'num_classes':4}

    net = create_model(opt)
    x = torch.autograd.Variable(torch.FloatTensor(1,3,size,size).uniform_(-1,1))

    out = net(x, out_feat_keys=net.all_feat_names)
    for f in range(len(out)):
        print('Output feature {0} - size {1}'.format(
            net.all_feat_names[f], out[f].size()))

    filters = net.get_L1filters()

    print('First layer filter shape: {0}'.format(filters.size()))
