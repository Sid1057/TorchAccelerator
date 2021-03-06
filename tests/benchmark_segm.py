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
input_16 = preprocess(input_image)
input_32 = preprocess(input_image)
input_16 = input_16.unsqueeze(0) # create a mini-batch as expected by the model
input_32 = input_32.unsqueeze(0) # create a mini-batch as expected by the model

# move the input and model to GPU
input_16 = input_16.cuda().half()
input_32 = input_32.cuda()

models = [
    (torchvision.models.segmentation.lraspp_mobilenet_v3_large, 'lraspp_mobilenetv3'),
    (torchvision.models.segmentation.deeplabv3_mobilenet_v3_large, 'deeplab movilenetv3'),
    (torchvision.models.segmentation.fcn_resnet50, 'fct resnet50')
]

modes = [
    '_fp32',
#    '_fp16',
#    '_fp32_trt',
#    '_fp16_trt',
#    '_fp16_trt_prune10',
#    '_fp16_trt_prune30',
#    '_fp16_trt_prune50',
]


for model, model_name in models:
    model_32 = model(pretrained=True).cuda().eval()
    # model_16 = model(pretrained=True).cuda().eval().half()

    print(f'==============================model: {model_name}==============================')
    for mode in modes:
        name = model_name + mode

        print('Model: ', name)

        trt_mode = 'trt' in name
        if trt_mode:
            print('use tensorrt inference engine')

        pruning_coef = 0
        if 'prune' in name:
            pruning_coef = float(name[-2:])/100
            print('pruning:', pruning_coef*100, '%')

        optimized_model = None
        input = None
        if '16' in name:
            input = input_16
            optimized_model = Model(model_16, input, trt_mode, pruning_coef)
        else:
            input = input_32
            optimized_model = Model(model_32, input, trt_mode, pruning_coef)

        with torch.no_grad():
            output = optimized_model(input)

            # probabilities = torch.nn.functional.softmax(output[0], dim=0).cpu().detach().numpy()
            # idx = np.argmax(probabilities)
            # print(f'{name} classification result: class {idx} with probability {probabilities[idx] :8f}')

            latency = optimized_model.profile(input, 100)
            fps = int(1000 / latency)

            print(f'fps: {fps}; latency: {latency} ms')

            print()

        torch.cuda.empty_cache()
        optimized_model = None
        input = None
    print()
