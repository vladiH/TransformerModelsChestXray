# import torch
import torch.nn as nn
import timm
class VitTransformerTimm(nn.Module):
    r""" Vit Transformer
        A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
           https://arxiv.org/abs/2010.11929v2

    Args:
        model_name (str): Pretrained timm model name getting from https://huggingface.co/timm/vit_tiny_patch16_224.augreg_in21k_ft_in1k. Default, vit_tiny_patch16_224.augreg_in21k_ft_in1k
        img_size (int | tuple(int)): Input image size. Default 224
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 14
        num_mlp_heads (int): Number of linear layers for each num_classes. Default 3
    """
    def __init__(self, model_name = 'vit_tiny_patch16_224.augreg_in21k_ft_in1k', img_size=224,  in_chans=3, num_classes=14, num_mlp_heads=3, **kwargs):
        super(VitTransformerTimm,self).__init__()
        self.model=timm.create_model(model_name, pretrained=True, in_chans=in_chans, num_classes=0, img_size=img_size)
        
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