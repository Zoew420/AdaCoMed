"""
单模态大模型embedding后的融合模型
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('./')
from LM.Image.dinov2.dinov2.hub.backbones import dinov2_vitg14_reg
from transformers import AutoTokenizer, AutoModel, AutoConfig
from peft import PeftModel
from llm2vec import LLM2Vec
import torchvision.transforms as transforms
from PIL import Image
import torch
import numpy as np

def preprocess_for_dinov2(image_tensor):
    # Assume image_tensor is [1, 128, 128]
    # Convert to PIL Image 
    image_np = image_tensor.squeeze().numpy()
    
    # Handle single channel or multi-channel images
    if image_np.ndim == 2:
        # Grayscale to RGB
        image_np = np.stack([image_np]*3, axis=-1)
    
    # Normalize to 0-255 range if not already
    if image_np.max() <= 1:
        image_np = (image_np * 255).astype(np.uint8)
    
    pil_image = Image.fromarray(image_np.astype(np.uint8))
    # Define transform explicitly
    dinov2_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Apply transform
    dinov2_input = dinov2_transform(pil_image)
    
    # Add batch dimension
    return dinov2_input

class ModalityExpert(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class MixtureOfModalityExperts(nn.Module):
    def __init__(self, hidden_dim, output_dim):
        super().__init__()
        
        # 定义 7 个专家，专家分别处理：Image Only, Text Only, Image+Text
        self.experts = nn.ModuleList([
            ModalityExpert(hidden_dim, hidden_dim, output_dim),      # Expert 1
            ModalityExpert(hidden_dim, hidden_dim, output_dim),      # Expert 2
            ModalityExpert(2 * hidden_dim, hidden_dim, output_dim),  # Expert 3
        ])
        
        # 门控网络，用于决定每个专家的权重
        self.gate = nn.Linear(2 * hidden_dim, 3)
        
    def forward(self, x):
        # 分割输入以对应不同的模态
        img_feat, text_feat = torch.split(x, x.size(1) // 2, dim=1)
        
        # 门控网络输入
        gate_input = torch.cat([img_feat, text_feat], dim=1)
        expert_weights = F.softmax(self.gate(gate_input), dim=-1)
        
        # 对每个专家应用相应的输入组合
        expert_outputs = [
            self.experts[0](img_feat),
            self.experts[1](text_feat),
            self.experts[2](torch.cat([img_feat, text_feat], dim=1)),
        ]
        
        # 使用门控权重组合专家输出
        expert_outputs = torch.stack(expert_outputs, dim=1)  # Shape: [batch_size, 3, output_dim]
        expert_weights = expert_weights.unsqueeze(2)  # Shape: [batch_size, 3, 1]
        output = torch.sum(expert_weights * expert_outputs, dim=1)  # Shape: [batch_size, output_dim]
        return output

class MultimodalMoETransformerBlock(nn.Module):
    def __init__(self, img_dim, text_dim, hidden_dim, num_heads, dropout=0.1):
        """
        将单模态大模型embedding后的特征融合
        目前：
        image embedding after Dinov2 [batch_size, 1536]
        text embedding after LLm2Vec [batch_size, 5120]
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # 各模态的投影层
        self.img_proj = nn.Linear(img_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # 多头自注意力机制
        self.self_attn = nn.MultiheadAttention(2 * hidden_dim, num_heads, dropout=dropout)
        
        # 混合模态专家替代传统的 MLP
        self.moe = MixtureOfModalityExperts(hidden_dim, 2*hidden_dim)
        
        # 层归一化和 dropout
        self.norm1 = nn.LayerNorm(2 * hidden_dim)
        self.norm2 = nn.LayerNorm(2 * hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, img, text):
        # 将每个模态投影到公共的隐藏维度
        # TODO: 这里的统一投影空间可以进行更复杂的设计
        img_proj_feat = self.img_proj(img)
        text_proj_feat = self.text_proj(text)
        
        # 连接投影后的特征
        x = torch.cat([img_proj_feat, text_proj_feat], dim=1)
        
        # 自注意力
        attn_output, _ = self.self_attn(x, x, x)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # 混合模态专家，使用处理后的特征
        moe_output = self.moe(x)
        x = x + self.dropout(moe_output)
        x = self.norm2(x)
        
        return x

class MultimodalMoEFusionNetwork(nn.Module):
    def __init__(self, img_dim=1536, text_dim=4096, hidden_dim=512, num_heads=8, dropout=0.1, device='cuda'):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.layer = MultimodalMoETransformerBlock(img_dim, text_dim, hidden_dim, num_heads, dropout)
        self.device = device
        
    def forward(self, img, text):
        img = img.to(self.device)
        text = text.to(self.device)
        x = self.layer(img, text)
        return x


class LargeOnlyTraining(nn.Module):
    def __init__(self, small_pretrained_model, img_dim=1536, text_dim=5120, device='cuda'):
        super(LargeOnlyTraining, self).__init__()
        self.small_pretrained_model = small_pretrained_model
        self.task_head = self.small_pretrained_model.classifier
        # 重新初始化这个classifier的权重
        for layer in self.task_head.children():
            if isinstance(layer, nn.Linear):  # 针对线性层重新初始化
                nn.init.xavier_uniform_(layer.weight)  # Xavier 初始化权重
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)  # 将偏置初始化为 0
            elif isinstance(layer, nn.Conv2d):  # 针对卷积层重新初始化（如有）
                nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
            elif isinstance(layer, nn.BatchNorm1d) or isinstance(layer, nn.BatchNorm2d):  # 针对 BN 层
                nn.init.ones_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.projection = nn.Linear(img_dim+text_dim, self.task_head.in_dim)
    
    def forward(self, img, text):
        feat = torch.cat([img, text], dim=1)
        feat = self.projection(feat)
        logits = self.task_head(feat)
        return logits
        

class ContrastiveCoTaskTraining(nn.Module):
    def __init__(self, small_pretrained_model, large_moe_model, hidden_dim=1024, weight_tpe='learnable', cls_dropout=0.1, loss_temperature=0.07, device='cuda', **weight_kwargs):
        super(ContrastiveCoTaskTraining, self).__init__()
        
        self.small_pretrained_model = small_pretrained_model
        
        self.large_moe_model = large_moe_model
        self.device = device
        self.cls_dropout = cls_dropout
        self.loss_temperature = loss_temperature
        
        self.weight_type = weight_tpe
        self.weight_kwargs = weight_kwargs
        # Freeze the small model's parameters except for the task_head
        for name, param in self.small_pretrained_model.named_parameters():
            if 'classifier' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        # Get the task head from the small pretrained model
        
        self.task_head = self.small_pretrained_model.classifier
        # Projection head for contrastive learning
        self.projection = nn.Linear(hidden_dim, self.task_head.in_dim)
        # Add learnable fusion weights
        self.fusion_weights = nn.Parameter(torch.ones(2))
        self.fusion_softmax = nn.Softmax(dim=0)
        

    def forward(self, large_image_embed, large_text_embed, small_feat):
        # Large model
        large_feat = self.large_moe_model(large_image_embed, large_text_embed)
        large_feat = self.projection(large_feat)
        # Task predictions
        large_logits = self.task_head(large_feat)
        small_logits = self.task_head(small_feat)
        
        # Projections for contrastive learning
        large_proj = F.normalize(large_feat, dim=1)
        small_proj = F.normalize(small_feat, dim=1)
        
        return large_logits, small_logits, large_proj, small_proj

    def contrastive_loss(self, proj1, proj2):
        # 首先归一化特征向量
        proj1 = F.normalize(proj1, p=2, dim=1)
        proj2 = F.normalize(proj2, p=2, dim=1)
        
        batch_size = proj1.shape[0]
        
        # 计算相似度矩阵
        sim = torch.mm(proj1, proj2.t()) / self.loss_temperature
        
        # 添加数值稳定性的处理
        sim_max, _ = torch.max(sim, dim=1, keepdim=True)  # 避免指数爆炸
        sim = sim - sim_max.detach()
        
        # 正样本对在对角线上
        pos = torch.diag(sim)
        
        # 计算所有负样本对
        exp_sim = torch.exp(sim)
        exp_pos = torch.exp(pos)
        
        # 移除对角线元素(正样本对)
        mask = ~torch.eye(batch_size, dtype=torch.bool, device=sim.device)
        neg = exp_sim * mask
        
        # 计算损失,添加小值避免log(0)
        eps = 1e-8
        frac = (exp_pos + eps) / (exp_pos + torch.mean(neg, dim=1) + eps)
        loss = -torch.log((exp_pos + eps) / (exp_pos + torch.sum(neg, dim=1) + eps))
        
        return frac, loss
        
    def inverse_weight(self, contrastive_loss, temperature=1.0, alpha=1.0, eps=1e-6):
        """
        - 当contrastive loss较大时，large_weight接近1
        - 当contrastive loss较小时，两个模型权重更均衡
        Args:
            contrastive_loss: 对比损失值
            temperature: 控制权重变化的平滑程度
            alpha: 控制权重变化的敏感度
            eps: 防止除零错误
        """
        # 计算基础权重
        scaled_loss = contrastive_loss / temperature
        inverse_loss = 1.0 / (alpha * scaled_loss + eps)
        
        # 归一化权重
        large_weight = torch.sigmoid(contrastive_loss)
        small_weight = inverse_loss / (inverse_loss + 1) # 归一化的倒数
        
        # 确保权重和为1
        total_weight = large_weight + small_weight
        large_weight = large_weight / total_weight
        small_weight = small_weight / total_weight
        
        return large_weight, small_weight
        
    def gaussian_weight(self, contrastive_loss, sigma=1.0):
        """
        - small_weight = exp(-contrastive_loss^2 / (2*sigma^2))
        - large_weight = 1 - small_weight
        
        Args:
            contrastive_loss: 对比损失值
            sigma: 高斯核的标准差，控制权重变化的平滑程度
        """
        small_weight = torch.exp(-contrastive_loss**2 / (2 * sigma**2))
        large_weight = 1 - small_weight
        return large_weight, small_weight
        
    def threshold_weight(self, contrastive_loss, threshold=0.5, slope=10):
        """
        - 当contrastive_loss < threshold时，权重更均衡
        - 当contrastive_loss > threshold时，large_logits权重快速增加
        
        Args:
            contrastive_loss: 对比损失值
            threshold: 切换点
            slope: 斜率，控制权重变化速度
        """
        x = contrastive_loss - threshold
        large_weight = 0.5 + 0.5 * torch.tanh(slope * x)
        # large_weight = torch.tanh(slope*x)
        small_weight = 1 - large_weight
        return large_weight, small_weight
    
    def learnable_weight(self):
        fusion_weights = self.fusion_softmax(self.fusion_weights)
        large_weight = fusion_weights[0]
        small_weight = fusion_weights[1]
        return large_weight, small_weight
                  
    def get_adaptive_weight(self, contrastive_loss):
        if self.weight_type == 'inverse':
            return self.inverse_weight(contrastive_loss, self.weight_kwargs.get('temperature', 1.0), self.weight_kwargs.get('alpha', 1.0), self.weight_kwargs.get('eps', 1e-6))
        elif self.weight_type == 'gaussian':
            return self.gaussian_weight(contrastive_loss, self.weight_kwargs.get('sigma', 1.0))
        elif self.weight_type == 'threshold':
            return self.threshold_weight(contrastive_loss, self.weight_kwargs.get('threshold', 2.0), self.weight_kwargs.get('slope', 10))
        elif self.weight_type == 'learnable':
            return self.learnable_weight()
        elif self.weight_type == 'ensemble':
            return 0.5, 0.5
        elif self.weight_type == 'small_domain':
            return 0.0, 1.0
        else:
            raise ValueError(f"Invalid weight type: {self.weight_type}")
        
        
class LM_embedding(object):
    def __init__(self, 
                image_model_path='LM/Image/dinov2_vitg14_reg4_pretrain.pth',
                text_model_path = 'LM/llm2vec-0.2.2/output/mntp/MeLLaMA-13B-chat-hf', 
                device = 'cuda'):
        
        """
        Load Image LM dinov2 model
        usage: self.image_model(img)
        output: [batch_size, 1536]
        """
        self.image_model = dinov2_vitg14_reg(pretrained=False)
        self.image_model.load_state_dict(torch.load(image_model_path))
        """
        Load Text LM model
        usage: self.text_model.encode(text)
        output: [batch_size, 5120]
        """
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_path)
        self.text_config = AutoConfig.from_pretrained(text_model_path)
        self.text_model = AutoModel.from_pretrained(text_model_path, trust_remote_code=True, config=self.text_config, torch_dtype=torch.bfloat16,
        device_map=device)
        self.text_model = PeftModel.from_pretrained(self.text_model, text_model_path)
        self.text_model = LLM2Vec(self.text_model, self.text_tokenizer, pooling_mode='mean', max_length=512)
        
        
        self.device = device
        
    def get_image_embedding(self, imgs):
        # Preprocess images for dinov2
        processed_imgs = torch.stack([preprocess_for_dinov2(img) for img in imgs])
        # Ensure float32 dtype
        processed_imgs = processed_imgs.float()
        
        self.image_model.to(self.device)
        processed_imgs = processed_imgs.to(self.device)
        
        # Potentially modify model to use float
        self.image_model = self.image_model.float()
        
        out_batch = self.image_model(processed_imgs)
        self.image_model.to('cpu')
        return out_batch
    
    def get_text_embedding(self, texts):
        def flatten_and_join(item):
            if isinstance(item, list):
                return ' '.join(flatten_and_join(subitem) for subitem in item)
            return str(item)
        def convert_lists_to_strings(texts):
            return [flatten_and_join(text) for text in texts]

        texts = convert_lists_to_strings(texts)
        self.text_model.to(self.device)
        
        # print(torch.cuda.memory_summary())
        text_reps = self.text_model.encode(texts, show_progress_bar=False)
        self.text_model.to('cpu')
        return text_reps

    
    def embedding(self, imgs, texts):
        # print("Getting image embedding...")
        with torch.no_grad():  # 不计算梯度，节省内存
            image_embedding = self.get_image_embedding(imgs)
            torch.cuda.empty_cache()
        # print("Getting text embedding...")
        with torch.no_grad():
            text_embedding = self.get_text_embedding(texts)
            torch.cuda.empty_cache()
        
        return image_embedding, text_embedding
    