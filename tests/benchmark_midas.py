import torch
import torchvision
from PIL import Image
import numpy as np
import sys
from torchvision import transforms
import urllib

# sys.path.append('..')
from torch_accelerator import Model

import cv2
import torch
import urllib.request

url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)

use_large_model = False

midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").cuda().eval().half()

midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

if use_large_model:
    transform = midas_transforms.default_transform
else:
    transform = midas_transforms.small_transform

img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

input_batch = transform(img).cuda().half()

optimized_model = Model(midas, input_batch, True, 0.0)

with torch.no_grad():
    optimized_model.profile(input_batch, 20)

    tt = start.elapsed_time(end)/100 / 10**3
    fps = int(1000 / tt)
    print('midas fps:', fps)
    print('midas latency:', tt)

    prediction = torch.nn.functional.interpolate(
        prediction.unsqueeze(1),
        size=img.shape[:2],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    
output = prediction.cpu().numpy()


