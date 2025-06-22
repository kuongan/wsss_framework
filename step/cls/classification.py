# Training Classification Model

import os
from tqdm import tqdm

import torch
from torch import nn, multiprocessing
import torch.distributed as dist

# from torchvision.datasets import VOCSegmentation, VOCDetection
from utils.datasets import voc_train_dataset, voc_val_dataset, voc_test_dataset
from utils.datasets import coco_train_dataset, coco_val_dataset
from torch.utils.data import DataLoader
# from torch.utils.data.distributed import DistributedSampler

import utils.loss
from utils.models import get_model
from utils.optims import get_cls_optimzier

from utils.misc import TensorBoardLogger, make_logger
from utils.train import validate, eval_multilabel_metric

import logging
logger = logging.getLogger('main')

# Seed (reproducibility)
    # import random
    # random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # cudnn.deterministic = True

def _work(pid, args, dataset_train, dataset_val, dataset_train_ulb):
    #dist.init_process_group(backend=args.distributed_backend, init_method=args.distributed_url,
    #                            world_size=1, rank=0)
    logger, _ = make_logger(args, is_new=False)

    # Initialize Tensorboard logger
    if args.use_tensorboard:
        tb_logger = TensorBoardLogger(args.log_path)

    # Dataloader
    #train_sampler = DistributedSampler(dataset_train)
    #val_sampler = DistributedSampler(dataset_val)
    train_dl = DataLoader(dataset_train, batch_size=args.train['batch_size'], num_workers=args.num_workers, 
                          shuffle=True, sampler=None, pin_memory=True)
    val_dl = DataLoader(dataset_val, batch_size=args.eval['batch_size'], num_workers=args.num_workers, 
                        shuffle=False, sampler=None, pin_memory=True)
    
    # Unlabeled dataloader
    if args.labeled_ratio < 1.0 or args.train_ulb_list:
        train_ulb_dl = DataLoader(dataset_train_ulb, batch_size=args.train['batch_size'],
                                  num_workers=args.num_workers, shuffle=True, sampler=None, pin_memory=True)

    # Get Model
    model = get_model(args.network, pretrained=True, num_classes=args.voc_class_num-1)
    
    # Optimizer
    optimizer, scheduler = get_cls_optimzier(args, model)
 
    # model dataparallel
    model = torch.nn.DataParallel(model).cuda()
    # model DDP(TBD)
    #model = torch.nn.parallel.DistributedDataParallel(model.cuda())

    # Loss (MultiLabelSoftMarginLoss or BCEWithLogitsLoss or etc..)
    criterion = getattr(utils.loss, args.train['loss']['name'])(**args.train['loss']['kwargs']).cuda()

    # Training 
    best_acc = 0.0
    for e in range(args.train['epochs']):
        tb_dict = {}
        # Validation
        if e % args.verbose_interval == 0:
            tb_dict['eval/loss'], tb_dict['eval/acc'], tb_dict['eval/precision'], \
                                tb_dict['eval/recall'], val_ap, tb_dict['eval/map'] = validate(model, val_dl, dataset_val, criterion)

            # Save Best Model
            if tb_dict['eval/acc'] >= best_acc:
                best_model_path = os.path.join(args.log_path, 'best.pth')
                torch.save(model.module.state_dict(), best_model_path)
                best_acc = tb_dict['eval/acc']

                logger.info(f'{best_model_path} Saved.')

            logger.info('Validation Loss: %.6f, mAP: %.2f, Accuracy: %.2f, Precision: %.2f, Recall: %.2f' % 
                        (tb_dict['eval/loss'], tb_dict['eval/map'], tb_dict['eval/acc'], tb_dict['eval/precision'], tb_dict['eval/recall']))

        model.train()
        train_loss = 0.
        logits, labels = [], []
        for img, label in tqdm(train_dl):
            # memorize labels
            labels.append(label)
            img, label = img.cuda(), label.cuda()
            
            # calc loss
            logit = model(img)            
            loss = criterion(logit, label).mean()

            # training
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # loss
            train_loss += loss.detach().cpu()
            # acc
            nth_logit = torch.sigmoid(logit).detach()
            logits.append(nth_logit)
            
        # Training log
        train_loss /= len(dataset_train)
        logits = torch.cat(logits, dim=0).cpu()
        labels = torch.cat(labels, dim=0) 
        
        # Logging
        tb_dict['train/acc'], tb_dict['train/precision'], tb_dict['train/recall'], ap, tb_dict['train/map'] = eval_multilabel_metric(labels, logits, average='samples')
        tb_dict['train/loss'] = train_loss
        tb_dict['train/classification_loss'] = train_loss
        tb_dict['lr'] = optimizer.param_groups[0]['lr'] # Need modification for other optims except SGDs

        logger.info('Epoch %d Train Loss: %.6f, mAP: %.2f, Accuracy: %.2f, Precision: %.2f, Recall: %.2f' % 
                    (e+1, tb_dict['train/loss'], tb_dict['train/map'], tb_dict['train/acc'], tb_dict['train/precision'], tb_dict['train/recall']))
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Update Tensorboard log
        tb_logger.update(tb_dict, e)

    # Final Validation
    tb_dict['eval/loss'], tb_dict['eval/acc'], tb_dict['eval/precision'], \
                                tb_dict['eval/recall'], val_ap, tb_dict['eval/map'] = validate(model, val_dl, dataset_val, criterion)
    logger.info('Final Validation Loss: %.6f, mAP: %.2f, Accuracy: %.2f, Precision: %.2f, Recall: %.2f' % 
                (tb_dict['eval/loss'], tb_dict['eval/map'], tb_dict['eval/acc'], tb_dict['eval/precision'], tb_dict['eval/recall']))
    # Logging
    tb_logger.update(tb_dict, args.train['epochs'])

    # Save final model (split module from dataparallel)
    final_model_path = os.path.join(args.log_path, 'final.pth')
    torch.save(model.module.state_dict(), final_model_path)
    logger.info(f'{final_model_path} Saved.')


def run(args):
    logger.info('Training Classifier...')

    # Count GPUs
    n_gpus = torch.cuda.device_count()
    logger.info(f'{n_gpus} GPUs Available.')

    # Dataset
    # VOC2012
    if args.dataset == 'voc12':
        # dataset
        dataset_train = voc_train_dataset(args, args.train_list, 'cls')
        dataset_val = voc_val_dataset(args, args.eval_list, 'cls')
        
        # Unlabeled dataset
        if args.labeled_ratio < 1.0 or args.train_ulb_list:
            dataset_train_ulb = voc_train_dataset(args, args.train_ulb_list, 'cls')
        else:
            dataset_train_ulb = None
    elif args.dataset == 'coco':
        dataset_train = coco_train_dataset(args, args.train_list, 'cls')
        dataset_val = coco_val_dataset(args, args.eval_list, 'cls')
        if args.labeled_ratio < 1.0 or args.train_ulb_list:
            dataset_train_ulb = coco_train_dataset(args, args.train_ulb_list, 'cls')
        else:
            dataset_train_ulb = None
    else:
        pass
    # # COCO2014
    # elif args.dataset == 'coco':
    #     pass
    # # Cityscapes
    # elif args.dataset == 'cityscapes':
    #     pass

    logger.info(f'Train Dataset Length: {len(dataset_train)}')
    logger.info(f'Validation Dataset Length: {len(dataset_val)}')
    if dataset_train_ulb is not None:
        logger.info(f'Unlabeled Train Dataset Length: {len(dataset_train_ulb)}')
    
    # Multiprocessing (But 1 process)
    multiprocessing.spawn(_work, nprocs=1, args=(args, dataset_train, dataset_val, dataset_train_ulb), join=True)
    
    logger.info('Done Finetuning.\n')

    return None
