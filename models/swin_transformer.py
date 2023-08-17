# import torch
import torch.nn as nn
import timm
from timm.models import SwinTransformer
class SwinTransformerTimm(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        model_name (str): Pretrained timm model name getting from https://huggingface.co/timm/swin_tiny_patch4_window7_224.ms_in1k. Default, swin_tiny_patch4_window7_224.ms_in1k.
        img_size (int | tuple(int)): Input image size. Default 224
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 14
        num_mlp_heads (int): Number of linear layers for each num_classes. Default 3
    """
    def __init__(self, img_size=224, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, num_mlp_heads=3, **kwargs):
        super(SwinTransformerTimm,self).__init__()
        self.model=SwinTransformer( img_size = img_size,
                                    patch_size = patch_size,
                                    in_chans = in_chans,
                                    num_classes = 0,
                                    embed_dim = embed_dim,
                                    depths = depths,
                                    num_heads = num_heads,
                                    window_size = window_size,
                                    mlp_ratio = mlp_ratio,
                                    qkv_bias = qkv_bias,
                                    qk_scale = qk_scale,
                                    drop_rate = drop_rate,
                                    drop_path_rate = drop_path_rate,
                                    ape = ape,
                                    patch_norm = patch_norm,
                                    use_checkpoint = use_checkpoint,
                                    num_mlp_heads = num_mlp_heads)
        
        #expected_img_size = self.model.default_cfg['input_size'][1]
        #assert img_size == expected_img_size, f"Input img_size ({img_size}) has the wrong value. Expected ({expected_img_size}) for the current model {model_name}."

        self.num_classes = num_classes
        self.num_mlp_heads = num_mlp_heads
        self.num_features = self.model.num_features

        self.heads = nn.ModuleList()
        if num_mlp_heads > 0:
            self.heads2 = nn.ModuleList()
        if num_mlp_heads > 1:
            self.heads3 = nn.ModuleList()
        if num_mlp_heads > 2:
            self.heads4 = nn.ModuleList()
        if num_mlp_heads > 0:
            self.relu = nn.ReLU()     # for 1 or more heads
        for _ in range(num_classes):
            if num_mlp_heads == 0:
                self.heads.append(nn.Linear(self.num_features, 2))
            if num_mlp_heads == 1:
                self.heads.append(nn.Linear(self.num_features, 48))
                self.heads2.append(nn.Linear(48, 2))
            if num_mlp_heads == 2:
                self.heads.append(nn.Linear(self.num_features, 384))
                self.heads2.append(nn.Linear(384, 48))
                self.heads3.append(nn.Linear(48, 2))
            if num_mlp_heads == 3:
                self.heads.append(nn.Linear(self.num_features, 384))
                self.heads2.append(nn.Linear(384, 48))
                self.heads3.append(nn.Linear(48, 48))
                self.heads4.append(nn.Linear(48, 2))
    
    def forward(self,x):
        x=self.model(x)
        y = []
        for i in range(len(self.heads)):
            if self.num_mlp_heads == 0:
                y.append(self.heads[i](x))
            if self.num_mlp_heads == 1:
                y.append(self.heads2[i](self.relu(self.heads[i](x))))
            if self.num_mlp_heads == 2:
                y.append(self.heads3[i](self.relu(self.heads2[i](self.relu(self.heads[i](x))))))
            if self.num_mlp_heads == 3:
                y.append(self.heads4[i](self.relu(self.heads3[i](self.relu(self.heads2[i](self.relu(self.heads[i](x))))))))
        return y