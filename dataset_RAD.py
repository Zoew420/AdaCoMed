"""
This code is modified based on Jin-Hwa Kim's repository (Bilinear Attention Networks - https://github.com/jnhwkim/ban-vqa) by Xuan B. Nguyen
"""
from __future__ import print_function
import collections
import os
import json
import _pickle as cPickle
import numpy as np
import utils
import torch
from torch.utils.data import Dataset
import itertools
import warnings
import re
import torch.nn.functional as F
from torch.utils.data.dataloader import default_collate
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
COUNTING_ONLY = False
string_classes = str
numpy_type_map = {
    'float64': torch.DoubleTensor,
    'float32': torch.FloatTensor,
    'float16': torch.HalfTensor,
    'int64': torch.LongTensor,
    'int32': torch.IntTensor,
    'int16': torch.ShortTensor,
    'int8': torch.CharTensor,
    'uint8': torch.ByteTensor,
}


def trim_collate(batch):
    "Puts each data field into a tensor with outer dimension batch size"
    _use_shared_memory = True
    error_msg = "batch must contain tensors, numbers, dicts or lists; found {}"
    elem_type = type(batch[0])
    if torch.is_tensor(batch[0]):
        out = None
        if 1 < batch[0].dim(): # image features
            max_num_boxes = max([x.size(0) for x in batch])
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = len(batch) * max_num_boxes * batch[0].size(-1)
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            # warning: F.pad returns Variable!
            return torch.stack([F.pad(x, (0,0,0,max_num_boxes-x.size(0))).data for x in batch], 0, out=out)
        else:
            if _use_shared_memory:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = batch[0].storage()._new_shared(numel)
                out = batch[0].new(storage)
            return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if re.search('[SaUO]', elem.dtype.str) is not None:
                raise TypeError(error_msg.format(elem.dtype))

            return torch.stack([torch.from_numpy(b) for b in batch], 0)
        if elem.shape == ():  # scalars
            py_type = float if elem.dtype.name.startswith('float') else int
            return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
    elif isinstance(batch[0], int):
        return torch.LongTensor(batch)
    elif isinstance(batch[0], float):
        return torch.DoubleTensor(batch)
    elif isinstance(batch[0], string_classes):
        return batch
    elif isinstance(batch[0], collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in batch[0]}
    elif isinstance(batch[0], collections.abc.Sequence):
        transposed = zip(*batch)
        return [trim_collate(samples) for samples in transposed]

    raise TypeError((error_msg.format(type(batch[0]))))

# Following Trott et al. (ICLR 2018)
#   Interpretable Counting for Visual Question Answering
def is_howmany(q, a, label2ans):
    if 'how many' in q.lower() or \
       ('number of' in q.lower() and 'number of the' not in q.lower()) or \
       'amount of' in q.lower() or \
       'count of' in q.lower():
        if a is None or answer_filter(a, label2ans):
            return True
        else:
            return False
    else:
        return False

def answer_filter(answers, label2ans, max_num=10):
    for ans in answers['labels']:
        if label2ans[ans].isdigit() and max_num >= int(label2ans[ans]):
            return True
    return False

class Dictionary(object):
    def __init__(self, word2idx=None, idx2word=None):
        if word2idx is None:
            word2idx = {}
        if idx2word is None:
            idx2word = []
        self.word2idx = word2idx
        self.idx2word = idx2word

    @property
    def ntoken(self):
        return len(self.word2idx)

    @property
    def padding_idx(self):
        return len(self.word2idx)

    def tokenize(self, sentence, add_word):
        sentence = sentence.lower()
        if "? -yes/no" in sentence:
            sentence = sentence.replace("? -yes/no", "")
        if "? -open" in sentence:
            sentence = sentence.replace("? -open", "")
        if "? - open" in sentence:
            sentence = sentence.replace("? - open", "")
        sentence = sentence.replace(',', '').replace('?', '').replace('\'s', ' \'s').replace('...', '').replace('x ray', 'x-ray').replace('.', '')
        words = sentence.split()
        tokens = []
        if add_word:
            for w in words:
                tokens.append(self.add_word(w))
        else:
            for w in words:
                # if a word is not in dictionary, it will be replaced with the last word of dictionary.
                tokens.append(self.word2idx.get(w, self.padding_idx-1))
        return tokens

    def dump_to_file(self, path):
        cPickle.dump([self.word2idx, self.idx2word], open(path, 'wb'))
        print('dictionary dumped to %s' % path)

    @classmethod
    def load_from_file(cls, path):
        print('loading dictionary from %s' % path)
        word2idx, idx2word = cPickle.load(open(path, 'rb'))
        d = cls(word2idx, idx2word)
        return d

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

def _create_entry(img, data, answer):
    if None!=answer:
        answer.pop('image_name')
        answer.pop('qid')
    entry = {
        'qid' : data['qid'],
        'image_name'    : data['image_name'],
        'image'       : img,
        'question'    : data['question'],
        'answer'      : answer,
        'answer_type' : data['answer_type'],
        'question_type': data['question_type'],
        'phrase_type' : data['phrase_type']}
    return entry

def is_json(myjson):
  try:
    json_object = json.loads(myjson)
  except ValueError:
    return False
  return True

def _load_dataset(dataroot, name, img_id2val, label2ans):
    """Load entries

    img_id2val: dict {img_id -> val} val can be used to retrieve image or features
    dataroot: root path of dataset
    name: 'train', 'val', 'test'
    """
    data_path = os.path.join(dataroot, name + 'set.json')
    samples = json.load(open(data_path))
    samples = sorted(samples, key=lambda x: x['qid'])

    answer_path = os.path.join(dataroot, 'cache', '%s_target.pkl' % name)
    answers = cPickle.load(open(answer_path, 'rb'))
    answers = sorted(answers, key=lambda x: x['qid'])

    utils.assert_eq(len(samples), len(answers))
    entries = []
    for sample, answer in zip(samples, answers):
        utils.assert_eq(sample['qid'], answer['qid'])
        utils.assert_eq(sample['image_name'], answer['image_name'])
        img_id = sample['image_name']
        if not COUNTING_ONLY or is_howmany(sample['question'], answer, label2ans):
            entries.append(_create_entry(img_id2val[img_id], sample, answer))

    return entries

class VQAFeatureDataset(Dataset):
    def __init__(self, name, args, dictionary, dataroot='data', question_len=12):
        super(VQAFeatureDataset, self).__init__()
        self.args = args
        assert name in ['train', 'test']
        dataroot = args.RAD_dir
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)

        # End get the number of answer type class
        self.dictionary = dictionary

        # TODO: load img_id2idx
        self.img_id2idx = json.load(open(os.path.join(dataroot, 'imgid2idx.json')))

        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans)
        # load image data for MAML module
        if args.maml:
            # TODO: load images
            images_path = os.path.join(dataroot, 'images84x84.pkl')
            print('loading MAML image data from file: '+ images_path)
            self.maml_images_data = cPickle.load(open(images_path, 'rb'))
        # load image data for Auto-encoder module
        if args.autoencoder:
            # TODO: load images
            images_path = os.path.join(dataroot, 'images128x128.pkl')
            print('loading DAE image data from file: '+ images_path)
            self.ae_images_data = cPickle.load(open(images_path, 'rb'))
        # tokenization
        self.tokenize(question_len)
        self.tensorize()
        if args.autoencoder and args.maml:
            self.v_dim = args.feat_dim * 2
        else:
            self.v_dim = args.feat_dim

    def tokenize(self, max_length=12):
        """Tokenizes the questions.

        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_idx in embedding
        """
        for entry in self.entries:
            tokens = self.dictionary.tokenize(entry['question'], False)
            tokens = tokens[:max_length]
            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self.dictionary.padding_idx] * (max_length - len(tokens))
                tokens = tokens + padding
            utils.assert_eq(len(tokens), max_length)
            entry['q_token'] = tokens

    def tensorize(self):
        if self.args.maml:
            self.maml_images_data = torch.from_numpy(self.maml_images_data)
            self.maml_images_data = self.maml_images_data.type('torch.FloatTensor')
        if self.args.autoencoder:
            self.ae_images_data = torch.from_numpy(self.ae_images_data)
            self.ae_images_data = self.ae_images_data.type('torch.FloatTensor')
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry['q_token']))
            entry['q_token'] = question

            answer = entry['answer']
            if None!=answer:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['q_token']
        answer = entry['answer']
        answer_type = entry['answer_type']
        question_type = entry['question_type']
        phrase_type = entry['phrase_type']

        image_data = [0, 0]
        if self.args.maml:
            maml_images_data = self.maml_images_data[entry['image']].reshape(84*84)
            image_data[0] = maml_images_data
        if self.args.autoencoder:
            ae_images_data = self.ae_images_data[entry['image']].reshape(128*128)
            image_data[1] = ae_images_data
        if None!=answer:
            labels = answer['labels']
            scores = answer['scores']
            # # 打印调试信息
            # print(f"VQAFeatureDataset - Index: {index}")
            # print(f"Labels: {labels}")
            # print(f"Scores: {scores}")
            target = torch.zeros(self.num_ans_candidates, dtype=torch.float)
            
            if labels is not None:
                # Ensure labels and scores are converted correctly
                labels = labels.clone().detach() if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long)
                scores = scores.clone().detach() if torch.is_tensor(scores) else torch.tensor(scores, dtype=torch.float)
            
                target.scatter_(0, labels, scores)
            return image_data, question, target, answer_type, question_type, phrase_type

        else:
            return image_data, question, answer_type, question_type, phrase_type

    def __len__(self):
        return len(self.entries)

class VQARawDataset(Dataset):
    def __init__(self, name, args, dictionary, dataroot='data'):
        super(VQARawDataset, self).__init__()
        self.args = args
        assert name in ['train', 'test']
        
        dataroot = args.RAD_dir
        ans2label_path = os.path.join(dataroot, 'cache', 'trainval_ans2label.pkl')
        label2ans_path = os.path.join(dataroot, 'cache', 'trainval_label2ans.pkl')
        self.ans2label = cPickle.load(open(ans2label_path, 'rb'))
        self.label2ans = cPickle.load(open(label2ans_path, 'rb'))
        self.num_ans_candidates = len(self.ans2label)
        self.dictionary = dictionary
        
        self.img_id2idx = json.load(open(os.path.join(dataroot, 'imgid2idx.json')))
        self.entries = _load_dataset(dataroot, name, self.img_id2idx, self.label2ans)
        
        images_path = os.path.join(dataroot, 'images128x128.pkl')
        print('loading image data from file: ' + images_path)
        self.images_data = cPickle.load(open(images_path, 'rb'))
        self.images_data = torch.from_numpy(self.images_data).type('torch.FloatTensor')
        
        # Add this block to process answers consistently
        for entry in self.entries:
            answer = entry['answer']
            if answer is not None:
                labels = np.array(answer['labels'])
                scores = np.array(answer['scores'], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry['answer']['labels'] = labels
                    entry['answer']['scores'] = scores
                else:
                    entry['answer']['labels'] = None
                    entry['answer']['scores'] = None

            
    def __getitem__(self, index):
        entry = self.entries[index]
        question = entry['question']
        answer = entry['answer']
        answer_type = entry['answer_type']
        question_type = entry['question_type']
        phrase_type = entry['phrase_type']

        image_data = self.images_data[entry['image']].reshape(1, 128, 128)

        if answer is not None:
            labels = answer['labels']
            scores = answer['scores']
            # # 打印调试信息
            # print(f"VQARawDataset - Index: {index}")
            # print(f"Labels: {labels}")
            # print(f"Scores: {scores}")
            # Create a consistent target tensor
            target = torch.zeros(self.num_ans_candidates, dtype=torch.float)
            
            if labels is not None:
                # Ensure labels and scores are converted correctly
                labels = labels.clone().detach() if torch.is_tensor(labels) else torch.tensor(labels, dtype=torch.long)
                scores = scores.clone().detach() if torch.is_tensor(scores) else torch.tensor(scores, dtype=torch.float)
            
                target.scatter_(0, labels, scores)
            return image_data, question, target, answer_type, question_type, phrase_type
        else:
            return image_data, question, answer_type, question_type, phrase_type

    def __len__(self):
        return len(self.entries)

def unified_collate(batch):
    """
    Unified collate function to handle both VQAFeatureDataset and VQARawDataset.
    - Supports different `images` and `questions` formats.
    - Ensures `target`, `answer_type`, `question_type`, and `phrase_type` are aligned.
    """
    # Initialize separate containers for different fields
    images = [x[0] for x in batch]  # Image data
    questions = [x[1] for x in batch]  # Question data (could be raw strings)
    targets = [x[2] for x in batch]  # Target (must align)
    ans_types = [x[3] for x in batch]  # Answer types
    q_types = [x[4] for x in batch]  # Question types
    p_types = [x[5] for x in batch]  # Phrase types

    # Process images
    if isinstance(images[0], list):  # For VQAFeatureDataset (list of [maml_image, ae_image])
        max_len_maml = max(img[0].size(0) if torch.is_tensor(img[0]) else 0 for img in images)
        max_len_ae = max(img[1].size(0) if torch.is_tensor(img[1]) else 0 for img in images)

        maml_images = torch.stack([
            torch.cat([img[0], torch.zeros(max_len_maml - img[0].size(0))]) if torch.is_tensor(img[0]) else torch.zeros(max_len_maml)
            for img in images
        ])

        ae_images = torch.stack([
            torch.cat([img[1], torch.zeros(max_len_ae - img[1].size(0))]) if torch.is_tensor(img[1]) else torch.zeros(max_len_ae)
            for img in images
        ])

        images = [maml_images, ae_images]
    else:  # For VQARawDataset (single tensor of shape [1, 128, 128])
        images = torch.stack(images)

    # Skip processing raw string questions; keep them as lists
    if isinstance(questions[0], str):  # Raw strings
        pass  # Keep as is
    elif torch.is_tensor(questions[0]):  # Already tensorized
        questions = torch.stack(questions)
    elif isinstance(questions[0], list):  # List of tokenized questions
        max_len = max(len(q) for q in questions)
        questions = torch.stack([
            F.pad(torch.tensor(q, dtype=torch.long), (0, max_len - len(q))) for q in questions
        ])

    # Process targets (aligning dimensions)
    max_candidates = max(t.size(0) for t in targets)
    targets = torch.stack([
        F.pad(t, (0, max_candidates - t.size(0))) for t in targets
    ])

    # Other fields: `answer_type`, `question_type`, `phrase_type` are lists, so no processing needed
    return images, questions, targets, ans_types, q_types, p_types


class VQANPYDataset(Dataset):
    def __init__(self, split, data_path='data/data_RAD/feat_npy'):
        img_path = os.path.join(data_path, f"{split}_img.npy")
        text_path = os.path.join(data_path, f"{split}_text.npy")
        label_path = os.path.join(data_path, f"{split}_label.npy")
        small_feat = os.path.join(data_path, f"{split}_small.npy")
        ans_type_path = os.path.join(data_path, f"{split}_ans_type.npy")
        
        self.img = np.load(img_path)
        self.text = np.load(text_path)
        self.label = np.load(label_path)
        self.small_feat = np.load(small_feat)
        self.ans_type = np.load(ans_type_path)
        
        # 将数据转换为tensor
        self.img = torch.tensor(self.img, dtype=torch.float32)
        self.text = torch.tensor(self.text, dtype=torch.float32)
        self.label = torch.tensor(self.label, dtype=torch.float32)
        self.small_feat = torch.tensor(self.small_feat, dtype=torch.float32)
        self.ans_type = list(self.ans_type)
        
    def __len__(self):
        return len(self.label)
    
    def __getitem__(self, idx):
        return self.img[idx], self.text[idx], self.small_feat[idx], self.label[idx], self.ans_type[idx]
    