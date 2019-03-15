import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

class Upsample(nn.Module):
    def __init__(self, scale_factor, mode="nearest"):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)

class Yolo(nn.Module):
    def __init__(self, anchors, nC, img_size):
        super().__init__()
        self.anchors = torch.FloatTensor(anchors)
        self.nA = len(anchors)
        self.nC = nC
        self.img_size = img_size
    
    def forward(self, x, img_size, targets=None, var=None):
        bs, nG = x.shape[0], x.shape[-1]
        if self.img_size != img_size:
            self.create_grids(img_size, nG)
            # NOTE: here leave out cuda activate
        x = x.view(bs, self.nA, self.nC + 5, nG, nG).permute(0, 1, 3, 4, 2).contiguous() 
        # x y width height
        xy = torch.sigmoid(x[..., 0:2])
        wh = x[..., 2:4]

        if targets is not None:
            loss_mse = nn.MSELoss()
            loss_bce nn.BCEWithLogitsLoss()
            loss_ce = nn.CrossEntropyLoss()

            conf_ = x[..., 4]
            cls_ = x[..., 5:]

            txy, twh, mask, tcls = build_targets(targets, self.anchor_vec, self.nA, self.nC, nG)

            tcls = tcls[mask]
            # NOTE: leave cuda activate  
            nT = sum([len(x) for x in targets])
            nM = mask.sum().float()
            k = 1 # nM / bs
            if nM > 0:
                lxy = k * loss_mse(xy[mask], txy[mask])
                lwh = k * loss_mse(wh[mask], twh[mask])
                lcls = (k / 4) * loss_ce(cls_[mask], torch.argmax(tcls, 1))
            else:
                FT = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
                lxy, lwh, lcls, lconf = FT([0]), FT([0]), FT([0]), FT([0])
            
            lconf = (k * 64) * loss_bce(conf_, mask.float())

            loss = lxy + lwh + lconf + lcls
            return loss, loss.item(), lxy.itme(), lwh.item(), lconf.item(), lcls.item(), nT
        else:
            x[..., 0:2] = xy + self.grid_xy
            x[..., 2:4] = torch.exp(wh) * self.anchor_wh
            x[..., 4] = torch.sigmoid(p[..., 4])
            x[..., :4] *= self.stride
            return x.view(bs, -1, 5 + self.nC)


    def create_grids(self, img_size, nG):
        self.stride = img_size // nG
        # build xy offsets
        grid_x = torch.arange(nG).repeat((nG, 1)).view((1, 1, nG, nG)).float()
        grid_y = grid_x.permute(0, 1, 3, 2)
        self.grid_xy = torch.stack((grid_x, grid_y), 4)
        # build wh gains
        self.anchor_vec = self.anchors / self.stride
        self.anchor_wh = self.anchor_vec.view(1, self.nA, 1, 1, 2)


class EmptyLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return x

# In this function, we need output_filters to construct convolution layer
def create_modules(hyper_config, net_config):
    output_filters = [hyper_config.img_shape[-1]] # the input channels 
    module_list = nn.ModuleList()
    for i, net in enumerate(net_config):
        modules = nn.Sequential()
        if net['type'] == 'conv':
            filters = net['filters']
            kernel_size = net['kernel_size']
            padding = (kernel_size - 1) // 2 if net['padding'] else 0
            conv = nn.Conv2d(output_filters[-1], filters, kernel_size, net['stride'], padding, bias=not net['bn'])
            modules.add_module('conv_%d' % i, conv)
            if net['bn'] == 1:
                modules.add_module('batch_norm_%d' % i, nn.BatchNorm2d(filters))
            if net['act'] == 'leaky':
                modules.add_module('leaky_%d' % i, nn.LeakyReLU(0.1))
        
        elif net['type'] == 'maxpool':
            kernel_size = net[' kernel_size']
            stride = net['stride']
            if kernel_size == 2 and stride == 1:
                modules.add_module('padding_%d' % i, nn.ZeroPad2d((0, 1, 0, 1)))
            maxpool = nn.MaxPool2d(kernel_size, stride, (kernel_size-1)//2)
            modules.add_module("maxpool_%d" % i, maxpool)
        
        elif net['type'] == 'upsample':
            upsample = Upsample(scale_factor=net('stride')) 
            modules.add_module('upsample_%d' % i, upsample)
        
        elif net['type'] == 'route':
            filters = sum(output_filters[i] for i in net['layers'])
            modules.add_module('shortcut_%d' % i, EmptyLayer())
        
        elif net['type'] == 'shortcut':
            filters = output_filters[net['from']]
            modules.add_module('shortcut_%d' % i, EmptyLayer())
        
        elif net['type'] == 'yolo':
            yolo = Yolo(net['anchors'], net['classes'], hyper_config.img_shape[0])
            modules.add_module('yolo_%d' % i, yolo)

    module_list.append(modules)
    # some layers modify filters but some not.
    output_filters.append(filters)
    return module_list

class Darknet(nn.Module):
    def __init__(self, image_size=416, channels=3):
        super().__init__()

if __name__ == '__main__':
    net = Darknet()
    print(net)