import argparse
import numpy as np
import os
import random
import torch



def fix_seed(random_seed):
    """
    fix seed to control any randomness from a code 
    (enable stability of the experiments' results.)
    """
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

# checkpoint
def save_checkpoint(checkpoint, saved_dir, file_name):
    os.makedirs(saved_dir, exist_ok=True) # make a directory to save a model if not exist.

    output_path = os.path.join(saved_dir, file_name)
    torch.save(checkpoint, output_path)


def load_checkpoint(checkpoint_path, model, optimizer, scheduler, rank=-1):
    # load model if resume_from is set
    
    if rank != -1: # distributed
        map_location = {"cuda:%d" % 0: 'cuda:%d' % rank}
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
    else:
        checkpoint = torch.load(checkpoint_path)    
        
    model.load_state_dict(checkpoint['model'])
    start_epoch = checkpoint['epoch']

    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    if scheduler is not None:
        scheduler.load_state_dict(checkpoint['scheduler'])

    return model, optimizer, scheduler, start_epoch

# TO-DO: option parse하는 부분 구현

def parse_opt():
    parser = argparse.ArgumentParser(description="Arguments for SimCLR.")

    # training
    parser.add_argument("--batch_size", type=int, default=2048)
    parser.add_argument("--optimizer", type=str, default="LARS")
    parser.add_argument("--lr", type=float, default=0.3)
    parser.add_argument("--eval_lr", type=float, default=0.1)
    parser.add_argument("--n_epochs", type=int, default=100)
    parser.add_argument("--acc_steps", type=int, default=2, help="number of steps for gradient accumulations") 
    parser.add_argument("--warmup_steps", type=int, default=10, help="the number of steps to apply linear warmup of a learning rate.")
    parser.add_argument("--linear_eval", action="store_true", help="A flag to train linear classifier on pre-training phase.")

    # augmentation
    parser.add_argument("--resize", type=int, default=224, help="a size after cropping an image.")
    parser.add_argument("--color_dist", type=float, default=1.0, help="a strength of color distortion.")

    # model
    parser.add_argument("--backbone", type=str, default="resnet50")
    parser.add_argument("--rep_dim", type=int, default=128)
    parser.add_argument("--return_h", type=bool, default=False, help="whether to return representation as well as z embedding.")
    parser.add_argument("--n_class", type=int, default=1000, help="number of classes for linear classification.")

    # optimizer
    parser.add_argument("--b1", type=float, default=0.9)
    parser.add_argument("--b2", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=1e-6)

    # loss
    parser.add_argument("--temp", type=float, default=0.1, help="a temperature value for NT-Xent Loss.")

    # dataset
    parser.add_argument("--root", type=str, default="/home/data/ImageNet", help="a root for data.")

    # misc
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb logging.")
    parser.add_argument("--prj_name", type=str, default="simclr", help="wandb project name.")
    parser.add_argument("--exp_name", type=str, default="exp1", help="a name of wandb run.")
    parser.add_argument("--track_grad", action="store_true", help="whether to track gradients and weights.")
    parser.add_argument("--log_interval", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0, help="a random seed for reproducibility.")
    parser.add_argument("--ckpt_dir", type=str, default="weights/", help="a path for weights to be saved.")
    parser.add_argument("--resume_from", action="store_true")
    parser.add_argument("--last_ckpt_dir", type=str, default="", help="a path for weights to be loaded.")

    # multi-processing
    parser.add_argument("--multi_gpu", type=bool, default=True)
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--n_workers", type=int, default=8)
    parser.add_argument("--world_size", type=int, default=1, help="number of nodes.")
    parser.add_argument("--gpu_id", type=int, nargs='+', default=-1, help="an id of current device, as a default, set to -1 which stands for cpu.")
    parser.add_argument("--node_rank", type=int, default=0)
    parser.add_argument("--n_nodes", type=int, default=1)

    opt = parser.parse_args()

    return opt

class AverageMeter(object):
    def __init__(self):
        self.init()

    def init(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val*n
        self.count += n
        self.avg = self.sum / self.count

def topk_accuracy(pred, true, k=1):
    """
    - Args
        pred: (n,k)의 prediction matrix
        true: (n,1)의 true label vector
    """
    pred_topk = pred.topk(k, dim=1)[1]
    n_correct = torch.sum(pred_topk.squeeze() == true)

    return n_correct / len(true)