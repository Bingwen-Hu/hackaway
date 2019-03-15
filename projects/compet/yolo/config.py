class Hyper(object):
    def __init__(self, training=True):
        # data config
        self.classes = 5
        self.image_dir = '/home/mory/data/round1_train/restricted'
        self.label_dir = '/home/mory/data/round1_train/restricted_ann'
        self.names = []

        # training or testing config
        self.batch = 16 if training else 1
        self.subdivisions = 1
        self.img_shape = (416, 416, 3)
        
        self.mmomentum = 0.9
        self.decay = 0.0005
        self.angle = 0
        self.saturation = 1.5
        self.exposure = 1.5
        self.hue = 0.1

        self.lr = 0.001
        self.brun_in = 1000
        self.max_batches = 500200
        self.policy = 400000, 450000
        self.steps = self.policy
        self.scales = 0.1, 0.1


class Net(object):
    def __init__(self):
        self.yolov3 = [
            # input
            {'net': 'conv', 'filters': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            # first part -> 64
            {'net': 'conv', 'filters': 64, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'act': 'leaky', 'bn': 1},
            [
                {'net': 'conv', 'filters': 32, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'conv', 'filters': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'shortcut', 'from': -3},
            ] * 1, 
            # second part -> 128
            {'net': 'conv', 'filters': 128, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'act': 'leaky', 'bn': 1},
            [
                {'net': 'conv', 'filters': 64, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'conv', 'filters': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'shortcut', 'from': -3}
            ] * 2,
            # third part -> 256
            {'net': 'conv', 'filters': 256, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'act': 'leaky', 'bn': 1},
            [
                {'net': 'conv', 'filters': 128, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'conv', 'filters': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'shortcut', 'from': -3},
            ] * 8,
            # fourth part -> 512
            {'net': 'conv', 'filters': 512, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'act': 'leaky', 'bn': 1},
            [
                {'net': 'conv', 'filters': 256, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'conv', 'filters': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'shortcut', 'from': -3},
            ] * 8,
            # fifth part -> 1024
            {'net': 'conv', 'filters': 1024, 'kernel_size': 3, 'stride': 2, 'padding': 1, 'act': 'leaky', 'bn': 1},
            [
                {'net': 'conv', 'filters': 512, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'conv', 'filters': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'shortcut', 'from': -3},
            ] * 4,
            # first yolo layer: detect larger object
            [
                {'net': 'conv', 'filters': 512, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'conv', 'filters': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            ] * 3,
            {'net': 'conv', 'filters': 255, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'linear', 'bn': 0},
            {'net': 'yolo', 'anchors': [(116, 90), (156, 198), (373, 326)], 'classes': 80, 'jitter': .3, 'ignore_thresh': .7, 'truth_thresh': 1, 'random': 1},
            # second yolo layer: detect medium object
            {'net': 'route', 'layers': [-4]},
            {'net': 'conv', 'filters': 256, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            {'net': 'upsample', 'stride': 2},
            {'net': 'route', 'layers': [-1, 61]},
            [
                {'net': 'conv', 'filters': 256, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'conv', 'filters': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            ] * 3, 
            {'net': 'conv', 'filters': 255, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'linear', 'bn': 0},
            {'net': 'yolo', 'anchors': [(30, 61), (62, 45), (59, 119)], 'classes': 80, 'jitter': .3, 'ignore_thresh': .7, 'truth_thresh': 1, 'random': 1},
            # third yolo layer: detect small object
            {'net': 'route', 'layers': [-4]},
            {'net': 'conv', 'filters': 128, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            {'net': 'upsample', 'stride': 2},
            {'net': 'route', 'layers': [-1, 36]},
            [
                {'net': 'conv', 'filters': 128, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
                {'net': 'conv', 'filters': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            ] * 3, 
            {'net': 'conv', 'filters': 255, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'linear', 'bn': 0},
            {'net': 'yolo', 'anchors': [(10, 13), (16, 30), (33, 23)], 'classes': 80, 'jitter': .3, 'ignore_thresh': .7, 'truth_thresh': 1, 'random': 1},
        ]
        self.yolov3_tiny = [
            {'net': 'conv', 'filters': 16, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            {'net': 'maxpool', 'kernel_size': 2, 'stride': 2}
            {'net': 'conv', 'filters': 32, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            {'net': 'maxpool', 'kernel_size': 2, 'stride': 2}
            {'net': 'conv', 'filters': 64, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            {'net': 'maxpool', 'kernel_si2e': 2, 'stride': 2}
            {'net': 'conv', 'filters': 128, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            {'net': 'maxpool', 'kernel_si2e': 2, 'stride': 2}
            {'net': 'conv', 'filters': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            {'net': 'maxpool', 'kernel_si2e': 2, 'stride': 2}
            {'net': 'conv', 'filters': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            {'net': 'maxpool', 'kernel_si2e': 2, 'stride': 1}
            {'net': 'conv', 'filters': 1024, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},

            {'net': 'conv', 'filters': 256, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            {'net': 'conv', 'filters': 512, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            {'net': 'conv', 'filters': 255, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'linear', 'bn': 0},
            {'net': 'yolo', 'anchors': [(81, 82), (135, 169), (344, 319)], 'classes': 80, 'jitter': .3, 'ignore_thresh': .7, 'truth_thresh': 1, 'random': 1},

            {'net': 'route', 'layers': [-4]},
            {'net': 'conv', 'filters': 128, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            {'net': 'upsample', 'stride': 2}
            {'net': 'route', 'layers': [-1, 8]},
            {'net': 'conv', 'filters': 256, 'kernel_size': 3, 'stride': 1, 'padding': 1, 'act': 'leaky', 'bn': 1},
            {'net': 'conv', 'filters': 255, 'kernel_size': 1, 'stride': 1, 'padding': 1, 'act': 'linear', 'bn': 0},
            {'net': 'yolo', 'anchors': [(10, 14), (23, 27), (37, 58)], 'classes': 80, 'jitter': .3, 'ignore_thresh': .7, 'truth_thresh': 1, 'random': 1},
        ]

        self.yolov3 = self.flatten(self.yolov3)

    def flatten(self, lst):
        ret = []
        for i in lst:
            if type(i) == list:
                ret.extend(i)
            else:
                ret.append(i)
        return ret


if __name__ == "__main__":
    nets = ConfigNet()
    yolov3 = nets.yolov3
    for i, net in enumerate(yolov3):
        print(i, net['net'], net.get('filters'))
    
