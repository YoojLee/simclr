import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import *
from augmentation import *
from dataset import SimCLRDataset
from loss import NTXentLoss
from scheduler import WarmupCosineDecay
from utils import *

# distributed backend
import torch.distributed as dist
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from datetime import timedelta
from tqdm import tqdm
from importlib import import_module
import wandb

def main():
    opt = parse_opt()
    fix_seed(opt.seed) # for reproducibility

    ngpus_per_node = len(opt.gpu_id)
    opt.world_size = ngpus_per_node * opt.n_nodes

    torch.multiprocessing.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, opt))

def main_worker(local_rank, ngpus_per_node, opt):
    """
    Contains Training Loop
    """
    global best_top1
    best_top1 = 0.0

    opt.local_rank = local_rank # 생각해보니 이게 굳이 필요한가 싶긴 함.
    torch.cuda.set_device(opt.local_rank) # set cuda device

    print(f"==> Device {opt.local_rank} working...")
    
    # define world rank. When nnodes set to 1, local rank and world rank will be same.
    opt.rank = opt.node_rank * ngpus_per_node + opt.local_rank
    
    # dist.init_process_group에서는 world_size와 world_rank를 필요로 함.
    dist.init_process_group(
        backend=opt.dist_backend, # nccl: gpu 백엔드
        init_method=f'tcp://127.0.0.1:11203',
        world_size=opt.world_size,
        rank=opt.rank,
        timeout=timedelta(300)
    )

    # Model
    model = SimCLR(model=opt.backbone, projection_dim=opt.rep_dim, return_h=opt.return_h).to(opt.local_rank)
    classifier = LinearClassifier(model=opt.backbone, n_class=opt.n_class).to(opt.local_rank)
    batch_size = int(opt.batch_size / ngpus_per_node)
    n_workers = int(opt.n_workers / ngpus_per_node)

    # Wrap the model in a DDP context
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model) # convert nn.BatchNorm to SyncBatchnorm -> 이 부분때문에 문제가 생기는 것 같은데..

    model = DistributedDataParallel(model, device_ids = [opt.local_rank])
    classifier = DistributedDataParallel(classifier, device_ids = [opt.local_rank])

    if opt.rank == 0:
        wandb.init(project=opt.prj_name, name=opt.exp_name, entity="yoojlee", config=vars(opt))
        if opt.track_grad:
            wandb.watch(model, log='all', log_freq=opt.log_interval)
   
    
    # Loss Function
    criterion = NTXentLoss(opt.temp).to(opt.local_rank)
    criterion_c = nn.CrossEntropyLoss().to(opt.local_rank)

    # optimizer
    if opt.optimizer != 'LARS':
        # 이때의 learning rate은 다르게 변경해야할 수도!
        optimizer_class = getattr(import_module("torch.optim"), opt.optimizer)
        optimizer = optimizer_class(
            {'params': model.parameters()},
            {'params': classifier.parameters(), 'lr': opt.eval_lr},
             lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.weight_decay)
    
    else:

        from torchlars import LARS

        base_optimizer_repr = torch.optim.SGD(
            model.parameters(),
            lr=opt.lr, weight_decay=opt.weight_decay)
        base_optimizer_cls = torch.optim.SGD(
            classifier.parameters(), lr=opt.eval_lr, weight_decay=opt.weight_decay
        )

        optimizer_repr = LARS(optimizer=base_optimizer_repr)
        optimizer_cls = LARS(optimizer=base_optimizer_cls)

    # dataset
    transform = SimCLRTransform(resize=opt.resize, s=opt.color_dist)
    transform_eval = BaseTransform(crop=opt.resize)
    train_data = SimCLRDataset(opt.root, True, transform, transform_eval)
    val_data = SimCLRDataset(opt.root, False, transform, transform_eval)

    train_sampler, val_sampler = DistributedSampler(train_data), DistributedSampler(val_data)
    train_loader = DataLoader(train_data,
                              batch_size=batch_size,
                              shuffle=(train_sampler is None),
                              num_workers=n_workers,
                              pin_memory=True,
                              sampler=train_sampler,
                              drop_last=True
                            )

    val_loader = DataLoader(val_data, 
                            batch_size=batch_size, 
                            shuffle=(val_sampler is None), 
                            num_workers=n_workers, 
                            pin_memory=True, 
                            sampler=val_sampler)

    # scheduler
    total_steps = int(len(train_data) / (batch_size*opt.acc_steps)) * (opt.n_epochs - opt.warmup_steps)
    warmup_steps = int(len(train_data) / (batch_size*opt.acc_steps)) * opt.warmup_steps
    scheduler = WarmupCosineDecay(optimizer_repr, warmup_steps, total_steps)

    # resume from
    if opt.resume_from:
        model, classifier, optimizer_repr, optimizer_cls, scheduler, start_epoch = load_checkpoint(opt.last_ckpt_dir, model, classifier, optimizer_repr, optimizer_cls, scheduler, opt.local_rank)
    
    else:
        start_epoch = 0

    dist.barrier()
    for epoch in range(start_epoch, opt.n_epochs):
        train_sampler.set_epoch(epoch)
        
        
        optimizer_repr.zero_grad()
        optimizer_cls.zero_grad()

        _ = train(train_loader, model, classifier, criterion, criterion_c, optimizer_repr, optimizer_cls, scheduler, epoch, opt)
        
        acc_score, _ = validate(val_loader, model, classifier, criterion, criterion_c, epoch, opt)

        if (opt.rank==0) and (best_top1 < acc_score):
            best_top1 = acc_score

            print(f"Saving Weights at the accuracy of {round(best_top1, 4)}")
            save_checkpoint(
                {
                    'epoch': epoch,
                    'model': model.module.state_dict(),
                    'best_top1': best_top1,
                    'optimizer_repr': optimizer_repr.state_dict(),
                    'optimizer_cls': optimizer_cls.state_dict(),
                    'scheduler': scheduler.state_dict()
                }, os.path.join(opt.ckpt_dir, opt.exp_name), f"{opt.prj_name}_{epoch}_{round(best_top1, 4)}.pt"
            )
        
            print(f"Best Accuracy: {round(best_top1, 4)}")
        
        dist.barrier()

            
        torch.cuda.empty_cache()

    if opt.rank==0:
        wandb.run.finish()

    dist.destroy_process_group()

def train(train_loader, model, classifier, criterion, criterion_c, optimizer, optimizer_c, scheduler, epoch, opt):
    model.train()
    acc_score, running_loss, running_loss_c = AverageMeter(), AverageMeter(), AverageMeter()
    pbar = tqdm(enumerate(train_loader), total=len(train_loader))

    for step, (image, image_eval, label) in pbar:
        image = image.view(-1, 3, opt.resize, opt.resize).to(opt.local_rank)
        
        # forward
        output = model(image)

        sim = get_cosine_similarity(output)
        
        # generate labels
        y = []
        for i in range(0,sim.size(0),2): # 이거 이렇게 안하고 더 빨리할 수 있는 방법 없나..?
            y.append(i+1) 
            y.append(i)

        y = torch.tensor(y).to(opt.local_rank)

        loss = criterion(sim,y)
        loss /= opt.acc_steps

        # backward
        loss.backward()

        running_loss.update(loss.item()*opt.acc_steps, image.size(0))

        ## linear evaluation
        if opt.linear_eval:
            label = label.to(opt.local_rank)
            image_eval = image_eval.to(opt.local_rank)

            pred = classifier(model.module.get_representation(image_eval).detach()) # DDP로 wrapping되어 있어서 module이라고 명시해줘야함.
            c_loss = criterion_c(pred, label)

            c_loss /= opt.acc_steps

            # backward
            c_loss.backward()

            running_loss_c.update(c_loss.item()*opt.acc_steps, image_eval.size(0))
            acc_score.update(topk_accuracy(pred.clone().detach(), label).item(), image_eval.size(0))

        if (step+1) % opt.acc_steps == 0:

            optimizer.step()
            optimizer_c.step()
            scheduler.step()
            optimizer.zero_grad()
            optimizer_c.zero_grad()

            # logging
            dist.barrier()
            if (opt.rank == 0) and ((step+1) % opt.log_interval == 0):
                wandb.log(
                    {
                        "Training Loss": round(running_loss.avg, 4),
                        "Training Cls Loss": round(running_loss_c.avg, 4),
                        "Training Accuracy": round(acc_score.avg, 4),
                        "Learning Rate": optimizer.param_groups[0]['lr'],
                        "Learning Rate (eval)": optimizer_c.param_groups[0]['lr']
                    }
                )

                running_loss.init()
                running_loss_c.init()
                acc_score.init()

            description = f'Epoch: {epoch+1}/{opt.n_epochs} || Step: {(step+1)//opt.acc_steps}/{len(train_loader)//opt.acc_steps} || Training Loss: {round(running_loss.avg, 4)} || Training Accuracy: {round(acc_score.avg, 4)}'
            pbar.set_description(description) # set a progress bar description only under rank 0

    return running_loss.avg
            


def validate(val_loader, model, classifier, criterion, criterion_c, epoch, opt):
    model.eval()

    acc_score, running_loss, running_loss_c = AverageMeter(), AverageMeter(), AverageMeter()
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))

    with torch.no_grad():
        for step, (image, image_eval, label) in pbar:
            image = image.view(-1, 3, opt.resize, opt.resize).to(opt.local_rank)

            sim = get_cosine_similarity(model(image)) # sim: [2N, 2N]
            
            y = []

            for i in range(0,sim.size(0),2): # N iterations
                y.append(i+1)
                y.append(i)
        
            y = torch.tensor(y).to(opt.local_rank)

            loss = criterion(sim, y)
            running_loss.update(loss.item(), image.size(0))

            if opt.linear_eval:
                image_eval = image_eval.to(opt.local_rank)
                label = label.to(opt.local_rank)

                pred = classifier(model.module.get_representation(image_eval).detach())
                
                c_loss = criterion_c(pred, label)
                running_loss_c.update(c_loss.item(), image_eval.size(0))

                acc_score.update(topk_accuracy(pred.clone().detach(), label).item(), image_eval.size(0))

            description = f'Current Epoch: {epoch+1} || Validation Step: {step+1}/{len(val_loader)} || Validation Loss: {round(loss.item(), 4)} || Validation Accuracy: {round(acc_score.avg, 4)}'
            pbar.set_description(description)

    if opt.rank == 0:
        wandb.log(
            {
                'Validation Loss': round(running_loss.avg, 4),
                'Validation Cls Loss': round(running_loss_c.avg, 4),
                'Validation Accuracy': round(acc_score.avg, 4)
            }
        )

    return acc_score.avg, running_loss.avg

if __name__ == "__main__":
    main()