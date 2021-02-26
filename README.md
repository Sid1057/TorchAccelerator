# TorchAccelerator
PyTorch model wrapper to accelerate models

## How to use

``` python
    from torch_accelerator import Model

    # torch stuff

    optimized_model = Model(model, input, trt_mode, pruning_coef)
    # model: pytorch nn.module
    # input: unput sample
    # trt_mode: true if you want to use tensorrt engine inference (have to install torch2trt)
    # pruning_coef: (from 0.1 to 1.0)
```

## benchmark

### 2070super

```
==============================model: squeezenet1_0==============================
Model:  squeezenet1_0_fp32
Using fp32 mode
squeezenet1_0_fp32 classification result: class 258 with probability 0.905274
fps: 601

Model:  squeezenet1_0_fp16
Using fp16 mode
squeezenet1_0_fp16 classification result: class 258 with probability 0.903809
fps: 549

Model:  squeezenet1_0_fp32_trt
Using fp32 mode
use tensorrt inference engine
squeezenet1_0_fp32_trt classification result: class 258 with probability 0.905484
fps: 1828

Model:  squeezenet1_0_fp16_trt
Using fp16 mode
use tensorrt inference engine
squeezenet1_0_fp16_trt classification result: class 258 with probability 0.906738
fps: 1800

Model:  squeezenet1_0_fp16_trt_prune10
Using fp16 mode
use tensorrt inference engine
pruning: 10.0 %
squeezenet1_0_fp16_trt_prune10 classification result: class 258 with probability 0.922363
fps: 1798

Model:  squeezenet1_0_fp16_trt_prune30
Using fp16 mode
use tensorrt inference engine
pruning: 30.0 %
squeezenet1_0_fp16_trt_prune30 classification result: class 258 with probability 0.850586
fps: 1825


==============================model: squeezenet1_1==============================
Model:  squeezenet1_1_fp32
Using fp32 mode
squeezenet1_1_fp32 classification result: class 258 with probability 0.930037
fps: 593

Model:  squeezenet1_1_fp16
Using fp16 mode
squeezenet1_1_fp16 classification result: class 258 with probability 0.929199
fps: 558

Model:  squeezenet1_1_fp32_trt
Using fp32 mode
use tensorrt inference engine
squeezenet1_1_fp32_trt classification result: class 258 with probability 0.930307
fps: 2973

Model:  squeezenet1_1_fp16_trt
Using fp16 mode
use tensorrt inference engine
squeezenet1_1_fp16_trt classification result: class 258 with probability 0.930176
fps: 2625

Model:  squeezenet1_1_fp16_trt_prune10
Using fp16 mode
use tensorrt inference engine
pruning: 10.0 %
squeezenet1_1_fp16_trt_prune10 classification result: class 258 with probability 0.938477
fps: 2754

Model:  squeezenet1_1_fp16_trt_prune30
Using fp16 mode
use tensorrt inference engine
pruning: 30.0 %
squeezenet1_1_fp16_trt_prune30 classification result: class 258 with probability 0.833984
fps: 2635


==============================model: resnet18==============================
Model:  resnet18_fp32
Using fp32 mode
resnet18_fp32 classification result: class 258 with probability 0.884896
fps: 595

Model:  resnet18_fp16
Using fp16 mode
resnet18_fp16 classification result: class 258 with probability 0.884277
fps: 521

Model:  resnet18_fp32_trt
Using fp32 mode
use tensorrt inference engine
resnet18_fp32_trt classification result: class 258 with probability 0.884760
fps: 1069

Model:  resnet18_fp16_trt
Using fp16 mode
use tensorrt inference engine
resnet18_fp16_trt classification result: class 258 with probability 0.885254
fps: 1070

Model:  resnet18_fp16_trt_prune10
Using fp16 mode
use tensorrt inference engine
pruning: 10.0 %
resnet18_fp16_trt_prune10 classification result: class 258 with probability 0.873535
fps: 1057

Model:  resnet18_fp16_trt_prune30
Using fp16 mode
use tensorrt inference engine
pruning: 30.0 %
resnet18_fp16_trt_prune30 classification result: class 258 with probability 0.775879
fps: 1042


==============================model: resnet50==============================
Model:  resnet50_fp32
Using fp32 mode
resnet50_fp32 classification result: class 258 with probability 0.873302
fps: 241

Model:  resnet50_fp16
Using fp16 mode
resnet50_fp16 classification result: class 258 with probability 0.873047
fps: 210

Model:  resnet50_fp32_trt
Using fp32 mode
use tensorrt inference engine
resnet50_fp32_trt classification result: class 258 with probability 0.873359
fps: 430

Model:  resnet50_fp16_trt
Using fp16 mode
use tensorrt inference engine
resnet50_fp16_trt classification result: class 258 with probability 0.873047
fps: 426

Model:  resnet50_fp16_trt_prune10
Using fp16 mode
use tensorrt inference engine
pruning: 10.0 %
resnet50_fp16_trt_prune10 classification result: class 258 with probability 0.875977
fps: 437

Model:  resnet50_fp16_trt_prune30
Using fp16 mode
use tensorrt inference engine
pruning: 30.0 %
resnet50_fp16_trt_prune30 classification result: class 258 with probability 0.824219
fps: 409

```