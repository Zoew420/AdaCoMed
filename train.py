import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import numpy as np
import pickle
import einops
import pprint
import transformers
import glob
import os
import sys
from tqdm import tqdm
from torch.cuda.amp import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve, auc
from sklearn.metrics import classification_report

from ml_collections import ConfigDict
from ml_collections.config_dict import config_dict
# 引入tensorboad 进行可视化
from torch.utils.tensorboard import SummaryWriter
import dataset_RAD
from model import *
from small_model import *
from datetime import datetime
import pandas as pd
import json
import argparse
import warnings
import random
from torchinfo import summary

warnings.filterwarnings("ignore")

use_amp = True

# 获取当前时间
now = datetime.now()

# 定义时间戳格式
train_timestamp = now.strftime('%Y-%m-%d_%H-%M-%S')
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

answer_types = ['CLOSED', 'OPEN', 'ALL']
quesntion_types = ['COUNT', 'COLOR', 'ORGAN', 'PRES', 'PLANE', 'MODALITY', 'POS', 'ABN', 'SIZE', 'OTHER', 'ATTRIB']
def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    return scores
    
def parse_args():
    parser = argparse.ArgumentParser(description='Large and MultimodalPred Model for Cotraining Cls')
    parser.add_argument('--task', type=str, default='VQA', help='task name')
    parser.add_argument('--co_model_name', type=str, default='ContrastiveCoTaskTraining', help='collaborative model name, ContrastiveCoTaskTraining/Weighted_Co_Task_Training/InverseContrastiveCoTaskTraining')
    parser.add_argument('--contrastive_loss_weight', type=float, default=0.5, help='contrastive loss weight')
    parser.add_argument('--large_image_model_name', type=str, default='dinov2')
    parser.add_argument('--large_text_model_name', type=str, default='MeLLaMA-13B')
    parser.add_argument('--resume', type=bool, default=False, help='resume')
    parser.add_argument('--resume_path', type=str, default='', help='resume path')
    parser.add_argument('--small_model_path', type=str, default='VQA/MICCAI19-MedVQA-master/saved_models/SAN_MEVF/model_epoch19.pth', help='small model path')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--epochs', type=int, default=100, help='epochs')
    parser.add_argument('--save_path', type=str, default='./saved_model', help='save path')
    parser.add_argument('--RAD_dir', type=str, default='data/data_RAD',
                        help='RAD dir')
    
    # 添加权重方案相关参数
    parser.add_argument('--weight_schemes', nargs='+', 
                        # default=['small_domain'],
                       default=['learnable', 'inverse', 'gaussian', 'threshold'],
                       help='List of weight schemes to try')
    
    # Inverse权重参数
    parser.add_argument('--inverse_temperature', type=float, default=1.0)
    parser.add_argument('--inverse_alpha', type=float, default=1.0)
    parser.add_argument('--inverse_eps', type=float, default=1e-6)
    
    # Gaussian权重参数
    parser.add_argument('--gaussian_sigma', type=float, default=1.0)
    
    # Threshold权重参数
    parser.add_argument('--threshold_value', type=float, default=0.95)
    parser.add_argument('--threshold_slope', type=float, default=10)
    
    # 模型通用参数
    parser.add_argument('--cls_dropout', type=float, default=0.1)
    parser.add_argument('--loss_temperature', type=float, default=0.07)
    
    # small model参数
    # Choices of Attention models
    parser.add_argument('--model', type=str, default='SAN', choices=['BAN', 'SAN'],
                        help='the model we use')

    # Choices of RNN models
    parser.add_argument('--rnn', type=str, default='LSTM', choices=['LSTM', 'GRU'],
                        help='the RNN we use')

    # BAN - Bilinear Attention Networks
    parser.add_argument('--gamma', type=int, default=2,
                        help='glimpse in Bilinear Attention Networks')
    parser.add_argument('--use_counter', action='store_true', default=False,
                        help='use counter module')

    # SAN - Stacked Attention Networks
    parser.add_argument('--num_stacks', default=2, type=int,
                        help='num of stacks in Stack Attention Networks')

    # Utilities - gpu
    parser.add_argument('--gpu', type=int, default=0,
                        help='specify index of GPU using for training, to use CPU: -1')

    # Question embedding
    parser.add_argument('--op', type=str, default='c',
                        help='concatenated 600-D word embedding')

    # Joint representation C dimension
    parser.add_argument('--num_hid', type=int, default=1024,
                        help='dim of joint semantic features')

    # Activation function + dropout for classification module
    parser.add_argument('--activation', type=str, default='relu', choices=['relu'],
                        help='the activation to use for final classifier')
    parser.add_argument('--dropout', default=0.5, type=float, metavar='dropout',
                        help='dropout of rate of final classifier')

    # Train with RAD
    parser.add_argument('--use_RAD', action='store_true', default=True,
                        help='Using TDIUC dataset to train')
    # Optimization hyper-parameters
    parser.add_argument('--eps_cnn', default=1e-5, type=float, metavar='eps_cnn',
                        help='eps - batch norm for cnn')
    parser.add_argument('--momentum_cnn', default=0.05, type=float, metavar='momentum_cnn',
                        help='momentum - batch norm for cnn')

    # input visual feature dimension
    parser.add_argument('--feat_dim', default=64, type=int,
                        help='visual feature dim')

    # Auto-encoder component hyper-parameters
    parser.add_argument('--autoencoder', action='store_true', default=True,
                        help='End to end model?')
    parser.add_argument('--ae_model_path', type=str, default='pretrained_ae.pth',
                        help='the maml_model_path we use')

    # MAML component hyper-parameters
    parser.add_argument('--maml', action='store_true', default=True,
                        help='End to end model?')
    parser.add_argument('--maml_model_path', type=str, default='pretrained_maml.weights',
                        help='the maml_model_path we use')
    
    args = parser.parse_args()
    return args
    

def train_contra_epoch(epoch, model, large_embedding, device, loader, task_criterion, contrastive_weight, optimizer, scaler, args):
    model.train()
    total_loss = []
    total_large_task_loss = []
    total_small_task_loss = []
    total_cotra_loss = []
    total_score = 0
    nan_batches = 0
    sample_size = 0
    # for batch_idx, (large_batch, small_batch) in enumerate(zip(large_loader, small_loader)):
    for batch in loader:
        torch.cuda.empty_cache()
        with autocast(enabled=use_amp):
            large_image, large_text, small_feat, a, ans_type = batch
                
            # Move to device
            large_image = large_image.to(device)
            large_text = large_text.to(device)
            small_feat = small_feat.to(device)
            a = a.to(device)
            large_logits, small_logits, large_proj, small_proj = model(large_image, large_text, small_feat)
            
            if torch.isnan(large_logits).any() or torch.isinf(large_logits).any() or \
               torch.isnan(small_logits).any() or torch.isinf(small_logits).any():
                nan_batches += 1
                continue
            # Compute losses
            with torch.autograd.set_detect_anomaly(True):
                if model.weight_type == 'learnable':
                    large_weight, small_weight = model.learnable_weight()
                    logits = large_weight * large_logits + small_weight * small_logits
                    task_loss = task_criterion(logits.float(), a)
                    frac, contrastive_loss = model.contrastive_loss(large_proj, small_proj)
                else:
                    frac, contrastive_loss = model.contrastive_loss(large_proj, small_proj)
                    large_weight, small_weight = model.get_adaptive_weight(frac.mean())
                    task_loss = task_criterion(large_weight * large_logits + small_weight * small_logits, a)
                # loss = task_loss.mean() + contrastive_weight * contrastive_loss.mean()
                loss = task_loss.mean()
                # Calculate scores
                large_weight, small_weight = model.get_adaptive_weight(frac.mean())
                final_preds = large_weight * large_logits + small_weight * small_logits
                batch_score = compute_score_with_logits(final_preds, a.data).sum()
                
                # Backward pass
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
        # Update metrics
        sample_size += a.size(0)
        total_score += batch_score
        total_loss.append(loss * a.size()[0])
    
    if len(total_loss) == 0:
        return float('nan'), 0
    
    # Calculate averages
    avg_loss = torch.sum(torch.stack(total_loss)) / sample_size
    score = total_score / sample_size
    
    # print(f"score: {score:.4f}")
          
    if nan_batches > 0:
        print(f"Skipped {nan_batches} batches due to NaN or Inf values.")
        
    return avg_loss, score

@torch.no_grad()
def val_epoch(epoch, model, large_embedding, device, loader, task_criterion, args):
    model.eval()
    keys = ['count', 'real', 'true', 'real_percent', 'score', 'score_percent']
    result = dict((i, dict((j, 0.0) for j in keys)) for i in answer_types)
    question_types_result = dict((i, dict((j, dict((k, 0.0) for k in keys)) for j in quesntion_types)) for i in answer_types)
    
    total_loss = []
    sample_size = 0
    nan_batches = 0
    
    # for batch_idx, (large_batch, small_batch) in enumerate(zip(large_loader, small_loader)):
    for batch in loader:
        torch.cuda.empty_cache()
        with autocast(enabled=use_amp):
            large_image, large_text, small_feat, a, ans_type = batch
            # Move to device
            large_image = large_image.to(device)
            large_text = large_text.to(device)
            small_feat = small_feat.to(device)
            a = a.to(device)
            large_logits, small_logits, large_proj, small_proj = model(large_image, large_text, small_feat)
            
            if torch.isnan(large_logits).any() or torch.isinf(large_logits).any() or \
               torch.isnan(small_logits).any() or torch.isinf(small_logits).any():
                nan_batches += 1
                continue
            
            # Calculate fusion weights and combined logits
            frac, contrastive_loss  = model.contrastive_loss(large_proj, small_proj)
            large_weight, small_weight = model.get_adaptive_weight(frac.mean())
            final_preds = large_weight * large_logits + small_weight * small_logits
            
            # Calculate loss and score
            loss = task_criterion(final_preds.float(), a) / a.size()[0]
            batch_score = compute_score_with_logits(final_preds, a.data).sum().item()

            # Update metrics
            sample_size += a.size(0)
            total_loss.append(loss * a.size()[0])
            # 更新结果字典
            ans_type = ans_type[0]
            batch_size = a.size(0)
            result[ans_type]['count'] += batch_size
            result[ans_type]['true'] += batch_score
            result[ans_type]['real'] += a.sum().item()
            result['ALL']['count'] += batch_size
            result['ALL']['true'] += batch_score
            result['ALL']['real'] += a.sum().item()
    
    if len(total_loss) == 0:
        return float('nan'), {}, {}
    
    # Calculate average loss
    avg_loss = torch.sum(torch.stack(total_loss)) / sample_size
    
    # 计算每种类型的分数
    for ans_type in result:
        if result[ans_type]['count'] > 0:
            result[ans_type]['score'] = result[ans_type]['true'] / result[ans_type]['count']
            result[ans_type]['score_percent'] = round(result[ans_type]['score'] * 100, 1)
        else:
            result[ans_type]['score'] = 0.0
            result[ans_type]['score_percent'] = 0.0


    if nan_batches > 0:
        print(f"Skipped {nan_batches} batches due to NaN or Inf values.")
        
    return avg_loss, result

def train(args, n_epochs, model, large_embedding, device, train_loader, val_loader, criterion, optimizer, scaler, save_path='./saved_model'):
    best_score = 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    nan_epochs = 0
    patience = 0
    max_nan_tolerance = 3
    max_patience = 5
    
    for epoch_idx in tqdm(range(n_epochs)):
        # Training phase
        train_loss, train_score = train_contra_epoch(
            epoch_idx, model, large_embedding, device, train_loader,
            criterion, args.contrastive_loss_weight, optimizer, scaler, args
        )
        torch.cuda.empty_cache()
        print(f'Epoch [{epoch_idx + 1}/{n_epochs}] loss:{train_loss:.3f} train_score:{train_score:.3f}')
        
        # Check for NaN loss
        if torch.isnan(train_loss):
            nan_epochs += 1
            if nan_epochs > max_nan_tolerance:
                print(f"NaN loss occurred for {nan_epochs} consecutive epochs. Terminating training.")
                break
        else:
            # Validation phase
            nan_epochs = 0
            val_loss, val_result = val_epoch(
                epoch_idx, model, large_embedding, device, val_loader,
                criterion, args
            )
            torch.cuda.empty_cache()
            
            val_score = val_result['ALL']['score_percent']
            print(f'Validation loss:{val_loss:.3f} score:{val_score:.1f}%')
            
            # Learning rate scheduling
            scheduler.step()
            
            # Model checkpointing
            if val_score > best_score:
                patience = 0
                print(f'New best score: {best_score:.1f}% -> {val_score:.1f}%')
                best_score = val_score
                
                # Save model
                save_dir = os.path.join(save_path, f'{args.weight_type}_{train_timestamp}')
                os.makedirs(save_dir, exist_ok=True)
                
                # Save model weights
                model_path = os.path.join(save_dir, 'model.pth')
                save_dict = {
                    'model_state_dict': model.state_dict(),
                    'args': vars(args),
                    'epoch': epoch_idx,
                    'best_score': best_score
                }
                torch.save(save_dict, model_path)
                
                # Save validation results
                results_path = os.path.join(save_dir, 'val_results.json')
                with open(results_path, 'w') as f:
                    json.dump(val_result, f, indent=2)
               
                print(f'Model and results saved to {save_dir}')
            # else:
            #     patience += 1
            #     if patience > max_patience:
            #         print(f"No improvement for {patience} epochs. Early stopping.")
            #         break
    
    # Final evaluation
    print("\nPerforming final evaluation on best model...")
    best_model = model  # Use the existing model
    best_model.load_state_dict(torch.load(model_path)['model_state_dict'])
    
    final_results = {}
    for split_name, loader in [('train', train_loader), ('val', val_loader)]:
        _, results = val_epoch(
            -1, best_model, large_embedding, device, 
            loader, criterion, args
        )
        final_results[split_name] = {
            'overall': results
        }
    # Save final results
    final_results_path = os.path.join(save_path, f'final_results_{args.weight_type}_{train_timestamp}.json')
    with open(final_results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
        
    print("\nFinal Scores:")
    for split in final_results:
        print(f"\n{split.upper()}:")
        for ans_type in answer_types:
            score = final_results[split]['overall'][ans_type]['score_percent']
            print(f"{ans_type}: {score:.1f}%")
    
    return best_score

def get_train_loaders(args):
    dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))
    
    small_train_dset = dataset_RAD.VQAFeatureDataset('train', args, dictionary) 
    large_train_dset = dataset_RAD.VQARawDataset('train', args, dictionary)
    small_train_dset.entries = sorted(small_train_dset.entries, key=lambda x: x['qid'])
    large_train_dset.entries = sorted(large_train_dset.entries, key=lambda x: x['qid'])

    assert len(small_train_dset) == len(large_train_dset), "Datasets must have same length"
    for i, (small_entry, large_entry) in enumerate(zip(small_train_dset.entries, large_train_dset.entries)):
        assert small_entry['qid'] == large_entry['qid'], f"QID mismatch at index {i}"
        assert small_entry['answer'] == large_entry['answer'], f"Answer mismatch at index {i}"
    
    # Use a custom sampler to ensure matching batches
    # train_sampler = torch.utils.data.RandomSampler(small_train_dset)
    class CustomSampler(torch.utils.data.Sampler):
        def __init__(self, dataset):
            self.indices = list(range(len(dataset)))

        def __iter__(self):
            return iter(self.indices)

        def __len__(self):
            return len(self.indices)
    train_sampler = CustomSampler(small_train_dset)

    for idx in range(len(small_train_dset)):
        small_data = small_train_dset[idx]
        large_data = large_train_dset[idx]

        if not torch.equal(small_data[2], large_data[2]):
            print(f"Mismatch at index {idx}")
            break

    small_train_loader = DataLoader(
        small_train_dset, 
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset_RAD.unified_collate
    )
    
    large_train_loader = DataLoader(
        large_train_dset,
        batch_size=args.batch_size,
        sampler=train_sampler,  # Use the same sampler
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset_RAD.unified_collate
    )
    
    for batch_idx, (large_batch, small_batch) in enumerate(zip(large_train_loader, small_train_loader)):
        large_target = large_batch[2]
        small_target = small_batch[2]
        for i in range(len(large_target)):
            if not torch.equal(large_target[i], small_target[i]):
                print(f"{i} in batch {batch_idx}")
                break

        
    return {'large': large_train_loader, 'small': small_train_loader}

def get_test_loaders(args):
    # Similar modification for test loaders
    dictionary = dataset_RAD.Dictionary.load_from_file(os.path.join(args.RAD_dir, 'dictionary.pkl'))
    
    small_test_dset = dataset_RAD.VQAFeatureDataset('test', args, dictionary) 
    large_test_dset = dataset_RAD.VQARawDataset('test', args, dictionary)
    assert len(small_test_dset) == len(large_test_dset), "Datasets must have same length"
    
    small_test_dset.entries = sorted(small_test_dset.entries, key=lambda x: x['qid'])
    large_test_dset.entries = sorted(large_test_dset.entries, key=lambda x: x['qid'])

    assert len(small_test_dset) == len(large_test_dset), "Datasets must have same length"
    for i, (small_entry, large_entry) in enumerate(zip(small_test_dset.entries, large_test_dset.entries)):
        assert small_entry['qid'] == large_entry['qid'], f"QID mismatch at index {i}"
        assert small_entry['answer'] == large_entry['answer'], f"Answer mismatch at index {i}"
        
    assert len(small_test_dset) == len(large_test_dset), "Datasets must have same length"
    
    small_test_loader = DataLoader(
        small_test_dset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset_RAD.unified_collate
    )
    
    large_test_loader = DataLoader(
        large_test_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=dataset_RAD.unified_collate
    )
    
    for batch_idx, (large_batch, small_batch) in enumerate(zip(large_test_loader, small_test_loader)):
        large_target = large_batch[2]
        small_target = small_batch[2]
        for i in range(len(large_target)):
            if not torch.equal(large_target[i], small_target[i]):
                print(f"{i} in batch {batch_idx}")
                break
    
    return {'large': large_test_loader, 'small': small_test_loader}

    
def get_npy_loaders(args):
    train_dset = dataset_RAD.VQANPYDataset('train')
    test_dset = dataset_RAD.VQANPYDataset('test')
    
    train_loader = DataLoader(
        train_dset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    
    
    test_loader = DataLoader(
        test_dset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    return train_loader, test_loader
    

def main(args):    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Setting device:", device)
    
    print("Loading Dataset")
    
    # Load data
    # train_loaders = get_train_loaders(args)
    # val_loaders = get_test_loaders(args) 
    train_loader, val_loader = get_npy_loaders(args)
    
    print("Loading small model")
    small_model = build_SAN(None, args)
    small_model_data = torch.load(args.small_model_path)
    small_model.load_state_dict(small_model_data.get('model_state', small_model_data))
    small_model.to(device)
    
    
    print("Loading large model")
    # large_embedding = LM_embedding()  # Initialize the embedding model
    large_embedding = None
    large_moe_model = MultimodalMoEFusionNetwork(text_dim=5120)
    large_moe_model.to(device)
    
    
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean').to(device)
    
    # 创建结果记录字典
    for weight_type in args.weight_schemes:
        args.weight_type = weight_type
        print(f"\nTraining with {weight_type} weighting scheme...")
        
        # 根据权重类型设置相应的参数
        weight_kwargs = {}
        if weight_type == 'inverse':
            weight_kwargs = {
                'temperature': args.inverse_temperature,
                'alpha': args.inverse_alpha,
                'eps': args.inverse_eps
            }
        elif weight_type == 'gaussian':
            weight_kwargs = {
                'sigma': args.gaussian_sigma
            }
        elif weight_type == 'threshold':
            weight_kwargs = {
                'threshold': args.threshold_value,
                'slope': args.threshold_slope
            }
            
        model = ContrastiveCoTaskTraining(
            small_model, 
            large_moe_model,
            num_classes=2,
            weight_tpe=weight_type,
            cls_dropout=args.cls_dropout,
            loss_temperature=args.loss_temperature,
            device=device,
            **weight_kwargs
        )
        model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total number of trainable parameters: {total_params / 1e6:.2f}M")

        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
        # scaler = GradScaler(init_scale=1024.0, growth_interval=2000, growth_factor=2.0)
        scaler = GradScaler()
        # 训练模型
        save_path = os.path.join(
            args.save_path, 
            f'{args.large_image_model_name}_{args.large_text_model_name}_VQA',
            args.co_model_name,
            weight_type
        )
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
            # Train
        best_score = train(
            args=args,
            n_epochs=args.epochs,
            model=model,
            large_embedding=large_embedding,
            device=device,
            train_loader=train_loader,
            val_loader=val_loader, 
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            save_path=save_path
        )
        
        print(f"Training complete. Best validation score: {best_score:.1f}%")
        
if __name__ == '__main__':
    args = parse_args()
    main(args)