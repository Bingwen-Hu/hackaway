# very simple config file

class Config():
    pass

config = Config()

# captcha property settings
config.charset = '12345678ABCDEFHJKMPRSTXY'
config.captlen = 6

# network setting
config.width = 140
config.height = 44
config.channel = 1

# training/testing setting
config.train_dir = ''
config.test_dir = ''
config.batch_size = 64
config.eval_step = 100
config.epochs = 100
config.model_path = 'best.pth'
config.baseline = 0.5

