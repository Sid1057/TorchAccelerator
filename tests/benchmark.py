import torch
import torchvision
from PIL import Image
import numpy as np
import sys
from torchvision import transforms
import urllib

# sys.path.append('..')
from torch_accelerator import Model

# download an image
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename)

# prepare input
input_image = Image.open(filename)
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input = preprocess(input_image)
input = input.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU
input = input.cuda()

models = [
    'squeezenet1_0',
    'squeezenet1_1',
    'resnet18',
    'resnet50',
    # 'resnet101',
    # 'resnet152',
    'vgg16'
]

modes = [
    '_fp32',
    '_fp16',
    '_fp32_trt',
    '_fp16_trt',
    '_fp16_trt_prune10',
    '_fp16_trt_prune30',
    # '_fp16_trt_prune50'
]


for model_name in models:
    model = torch.hub.load('pytorch/vision:v0.6.0', model_name, pretrained=True, verbose=False).cuda().eval()

    print(f'==============================model: {model_name}==============================')
    for mode in modes:
        name = model_name + mode

        print('Model: ', name)
        if '16' in name:
            print('Using fp16 mode')
            model = model.half()
            input = input.half()
        else:
            print('Using fp32 mode')
            model = model.float()
            input = input.float()

        trt_mode = 'trt' in name
        if trt_mode:
            print('use tensorrt inference engine')

        pruning_coef = 0
        if 'prune' in name:
            pruning_coef = float(name[-2:])/100
            print('pruning:', pruning_coef*100, '%')

        optimized_model = Model(model, input, trt_mode, pruning_coef)
        with torch.no_grad():
            output = optimized_model(input)

            probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().detach().numpy()
            idx = np.argmax(probabilities)
            print(f'{name} classification result: class {idx} with probability {probabilities[idx] :8f}')

            fps = int(1 / optimized_model.profile(input, 500))

            print(f'fps: {fps}')

            print()

        torch.cuda.empty_cache()
    print()
