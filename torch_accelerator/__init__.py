import torch
from torch2trt import torch2trt
import tensorrt
import time


class Model:
    def __init__(self, model, input, trt=True, prune_amount=0, verbose=False, force_fp16=False):
        self.__model = model

        if prune_amount >= 0.1:
            self.prune(prune_amount)

        if trt:
            fp16_mode = force_fp16
            if isinstance(input, torch.Tensor):
                fp16_mode = input.dtype==torch.float16 or force_fp16

            self.__model = torch2trt(self.__model, [input], fp16_mode=fp16_mode)

    def __call__(self, *args):
        return self.__model(*args)

    def prune(self, amount): 
    # Prune model to requested global sparsity 
        import torch.nn.utils.prune as prune
        for name, m in self.__model.named_modules(): 
            if isinstance(m, torch.nn.Conv2d): 
                prune.l1_unstructured(m, name='weight', amount=amount)  # prune 
                prune.remove(m, 'weight')  # make permanent

    def profile(self, input, experiments_count=100):
        ms = None

        with torch.no_grad():
            torch.cuda.current_stream().synchronize()
            t0 = time.time()

            for _ in range(experiments_count):
                output = self.__model(input) # JUST FOR WARMUP
                torch.cuda.current_stream().synchronize()
            t1 = time.time()

            ms = 1000 * (t1 - t0) / experiments_count

        return ms

    def info(self):
        return ''

    def __repr__(self):
        return ''

    def __repr__(self):
        return ''
