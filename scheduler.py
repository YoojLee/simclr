# 나중에 다시 구현하자..
from torch.optim.lr_scheduler import LambdaLR
import math

class WarmupCosineDecay(LambdaLR):
    """
    Linear warmup and then cosine decay.
    """
    def __init__(self, optimizer, warmup_steps, total_steps, cycle_factor=1., last_epoch=-1, verbose=False):
        """
        - Args
            optimizer: optimizer to adjust learning rate
            warmup_steps: steps for a learning rate warm-ups (linear warmup)
            total_steps: total steps
            cycle_factor: factor for adjusting a period T_i
        """
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.cycle_factor = cycle_factor # 주기에 적용하는 factor인 듯함.
        super(WarmupCosineDecay, self).__init__(optimizer, self.lr_lambda, last_epoch, verbose)
    
    def lr_lambda(self, step):
        # warm-up 적용
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        
        # min_eta = 0, max_eta = initial_lr로 가정. (그렇게 되니까 lr_lambda를 그냥 쓰는 거겠지 -> lr_lambda 자체가 lambda function으로 초기 학습률값을 조작하는 방식으로 동작)
        # 얘는 annealing이 아니라 그냥 전체 step을 하나의 주기로 보는 것 같음. annealing을 하나 더 구현해보자!
        progress = min(1, max(0,(step - self.warmup_steps) / float(self.total_steps - self.warmup_steps))) # clipping (progress를 0과 1 사이로 강제)
        return 0.5 * (1. + math.cos(math.pi*progress*float(self.cycle_factor)))
