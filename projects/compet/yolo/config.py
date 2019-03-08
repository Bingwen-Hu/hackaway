class ConfigBase(object):
    def __init__(self, training=True):
        # data config
        self.classes = 5
        self.image_dir = '/home/mory/data/round1_train/restricted'
        self.label_dir = '/home/mory/data/round1_train/restricted_ann'
        self.names = []

        # training or testing config
        self.batch = 16 if training else 1
        self.subdivisions = 1
        self.shape = (416, 416, 3)
        
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


