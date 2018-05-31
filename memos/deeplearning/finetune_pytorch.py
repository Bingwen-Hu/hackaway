  
import os  
import numpy as np  
  
import torch  
import torch.nn as nn
import torchvision.models as models  
from torch.autograd import Variable   
import torchvision.transforms as transforms  
  
from PIL import Image  
  
img_to_tensor = transforms.ToTensor()  
  
def make_model():  
    vgg16 = models.vgg16(pretrained=True)  
    return vgg16
  
      
#特征提取  
def extract_feature(vgg16, imgpath):
    vgg16.eval()
    img = Image.open(imgpath)  
    img = img.resize((224,224))
    tensor = img_to_tensor(img)  
      
    tensor = tensor.resize_(1,3,224,224)
              
    result = vgg16(Variable(tensor))
    result_npy = result.data.cpu().numpy()  
      
    return result_npy[0]  
      
class moryVGG16(nn.Module):

    def __init__(self, num_classes):
        super(moryVGG16, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
        )
        self.classifier = nn.Sequential(
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=num_classes, bias=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__=="__main__":  
    model = make_model()
    myvgg = moryVGG16(248)
    discard = ['classifier.6.weight', 'classifier.6.bias']
    pretrained_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k not in discard}
    mystate_dict = myvgg.state_dict()
    mystate_dict.update(pretrained_dict)
    myvgg.load_state_dict(mystate_dict)
    img = "E:/captcha-data/dwnews/test/aesa.jpg"
    feature = extract_feature(myvgg, img)
    print(len(feature))