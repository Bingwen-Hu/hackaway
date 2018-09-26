# very simple config file

class Config():
    pass

config = Config()

# captcha property settings
config.charset = '12345678ABCDEFHIJKMPRSTVXY'
config.charlen = len(config.charset)
config.captlen = 6

# network setting
config.width = 140
config.height = 140
config.channel = 3

# training/testing setting
config.train_dir = 'D:/Mory/sogou/train/'
config.test_dir = 'D:/Mory/sogou/test/'

config.epochs = 2
config.batch_size = 64

config.eval_step = 100
config.model_path = 'best.pth'
config.baseline = 0.5
config.restore = False
