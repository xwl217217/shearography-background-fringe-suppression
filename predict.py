import cv2
import numpy as np
from unet_model import UNet
import os
import torch
from torchvision import transforms
from normpi import normpi

tran = transforms.Compose([
    transforms.ToTensor(),
    normpi()
])
txtfilepath = 'testimgs'
imgfile = os.listdir(txtfilepath)  # 输入要预测的图片所在路径
imgfile.sort(key=lambda x: int(x[:-4]))
model = UNet(bilinear=True)
checkpoint = torch.load('./experiments/checkpoints/best.ckpt', map_location='cpu')
model.load_state_dict(checkpoint['model'])


def predict():
    torch.no_grad()
    for i in range(len(imgfile)):
        name = imgfile[i]
        filename = txtfilepath + '/' + name
        img = cv2.imdecode(np.fromfile(filename, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        # img = cv2.resize(img, (64, 64))  # 是否需要resize取决于新图片格式与训练时的是否一致
        img = img.reshape((*img.shape, -1))
        img = tran(img)
        img = img.unsqueeze(0)
        outputs = model(img)  # outputs，out1修改为你的网络的输出
        result = outputs.detach().numpy()
        os.makedirs("testresult", exist_ok=True)
        filename = os.path.join("testresult", f"{i + 1}.txt")
        np.savetxt(filename, result)


if __name__ == '__main__':
    predict()
