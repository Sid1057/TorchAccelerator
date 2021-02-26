import torch
from torch2trt import torch2trt
import tensorrt


class Model:
    def __init__(self, model, input, trt=True, prune_amount=0, verbose=False):
        self.__model = model

        if prune_amount >= 0.1:
            self.prune(prune_amount)

        if trt:
            self.__model = torch2trt(self.__model, [input])

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
        output = None
        with torch.no_grad():
            wTime = 0
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            output = self.__model(input) # JUST FOR WARMUP

            start.record()
            for i in range(experiments_count):
                got = self.__model(input)

            end.record()
            torch.cuda.synchronize()

            output = start.elapsed_time(end)/experiments_count / 10**3
            # print('execution time in SECONDS: {}'.format(output))

        return output

    def info(self):
        return ''

    def __repr__(self):
        return ''

    def __repr__(self):
        return ''
