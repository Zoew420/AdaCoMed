# 加载Raw数据集并用大模型处理过后存成NPY的
import argparse
import torch
from torch.utils.data import DataLoader
import dataset_RAD
import utils
import pandas as pd
import os
import json
from model import *
from small_model import *
from tqdm import tqdm

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ensemble', type=bool, default=False,
                        help='ensemble flag. If True, generate a logit file which is used in the ensemble part')
    # MODIFIABLE MEVF HYPER-PARAMETERS--------------------------------------------------------------------------------
    # Model loading/saving
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--input', type=str, default='saved_models/SAN_MEVF',
                        help='input file directory for loading a model')
    parser.add_argument('--output', type=str, default='results',
                        help='output file directory for saving VQA answer prediction file')
    # Utilities
    parser.add_argument('--epoch', type=int, default=19,
                        help='the best epoch')

    # Gradient accumulation
    parser.add_argument('--batch_size', type=int, default=16,
                        help='batch size')

    parser.add_argument('--RAD_dir', type=str, default='data/data_RAD',
                        help='RAD dir')
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
    
    parser.add_argument('--small_model_path', type=str, default='VQA/MICCAI19-MedVQA-master/saved_models/SAN_MEVF/model_epoch19.pth', help='small model path')

    # Return args
    args = parser.parse_args()
    return args

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

def get_embedding(loaders, large_embedding, small_model):
    image_embed = []
    text_embed = []
    small_feats = []
    targets = []
    ans_types = []
    for batch_idx, (large_batch, small_batch) in enumerate(tqdm(zip(loaders['large'], loaders['small']), desc="Processing Batches", total=min(len(loaders['large']), len(loaders['small'])))):
        torch.cuda.empty_cache()
        # Unpack batches
        # Process large batch embeddings
        large_images, large_questions, large_target, large_ans_types, large_q_types, large_p_types = large_batch
        large_image, large_text = large_embedding.embedding(large_images, large_questions)
        # large_image, large_text, large_target = large_batch
        
        v, q, a, ans_type, q_types, p_type = small_batch
        assert torch.equal(large_target, a), "Large and small batch targets do not match."
        
        if p_type[0] != "freeform":
            continue
            
        v[0] = v[0].reshape(v[0].shape[0], 84, 84).unsqueeze(1)
        v[1] = v[1].reshape(v[1].shape[0], 128, 128).unsqueeze(1)
        v[0] = v[0].to(device)
        v[1] = v[1].to(device)
        q = q.to(device)
        a = a.to(device)
        
        # Forward pass
        with torch.no_grad():
            small_feat, _ = small_model(v, q)
        image_embed.append(large_image.cpu().numpy())
        text_embed.append(large_text.cpu().numpy())
        targets.append(large_target.cpu().numpy())
        small_feats.append(small_feat.cpu().numpy())
        ans_types.append(large_ans_types)
    image_embed = np.concatenate(image_embed, axis=0)
    text_embed = np.concatenate(text_embed, axis=0)
    targets = np.concatenate(targets, axis=0)
    small_feats = np.concatenate(small_feats, axis=0)
    ans_types = np.concatenate(ans_types, axis=0)
    return image_embed, text_embed, small_feats, targets, ans_types


def save_embeddings(split, img_embs, text_embs, small_feat, label, ans_type, save_path):
    np.save(os.path.join(save_path, f"{split}_img.npy"), img_embs)
    np.save(os.path.join(save_path, f"{split}_text.npy"), text_embs)
    np.save(os.path.join(save_path, f"{split}_small.npy"), small_feat)
    np.save(os.path.join(save_path, f"{split}_label.npy"), label)
    np.save(os.path.join(save_path, f"{split}_ans_type.npy"), ans_type)
    

def main(save_dir='data/data_RAD/feat_npy'):
    args = parse_args()
    # Load data
    train_loaders = get_train_loaders(args)
    test_loaders = get_test_loaders(args) 
    
    print("Loading small model")
    print("train_loaders['small'].dataset.dictionary.ntoken", train_loaders['small'].dataset.dictionary.ntoken)
    print("train_loaders['small'].dataset.num_ans_candidates", train_loaders['small'].dataset.num_ans_candidates)
    print("train_loaders['small'].dataset.vdim", train_loaders['small'].dataset.v_dim)
    small_model = build_SAN(train_loaders['small'].dataset, args)
    small_model_data = torch.load(args.small_model_path)
    small_model.load_state_dict(small_model_data.get('model_state', small_model_data))
    small_model.to(device)
    
    large_embedding = LM_embedding()  # Initialize the embedding model
    train_img, train_txt, train_small, train_label, train_ans_type = get_embedding(train_loaders, large_embedding, small_model)
    save_embeddings('train', train_img, train_txt, train_small, train_label, train_ans_type, save_dir)
    test_img, test_txt, test_small, test_label, test_ans_type = get_embedding(test_loaders, large_embedding, small_model)
    save_embeddings('test', test_img, test_txt, test_small, test_label, test_ans_type, save_dir)
    
if __name__ == '__main__':
    main()