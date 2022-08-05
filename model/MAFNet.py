# coding:utf-8
# By Zhen Feng, Aug. 5, 2022
# Email: zfeng94@outlook.com

import math
import numpy as np
import torch
import torch.nn as nn
from torch.nn.modules.container import T 
import torchvision.models as models


def swish(x):
    return x * torch.sigmoid(x)
ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}

class MAFNet(nn.Module):

    def __init__(self, n_class,input_h,input_w):
        super(MAFNet, self).__init__()

        resnet_raw_model1 = models.resnet34(pretrained=True)
        resnet_raw_model2 = models.resnet34(pretrained=True)
        self.inplanes = 512
        self.input_h = input_h
        self.input_w = input_w

        ########  Disparity ENCODER  ########


        self.encoder_disparity_conv1 = resnet_raw_model1.conv1
        self.encoder_disparity_bn1 = resnet_raw_model1.bn1
        self.encoder_disparity_relu = resnet_raw_model1.relu
        self.encoder_disparity_maxpool = resnet_raw_model1.maxpool
        self.encoder_disparity_layer1 = resnet_raw_model1.layer1
        self.encoder_disparity_layer2 = resnet_raw_model1.layer2
        self.encoder_disparity_layer3 = resnet_raw_model1.layer3

        self.encoder_disparity_layer316 = Transformer(img_size=(input_h//16, input_w//16), patch_size=(2,2), in_channels=self.inplanes//2, hidden_size = self.inplanes*2, out_channels=self.inplanes, num_attention_heads = 16)  #res34

        ########  RGB ENCODER  ########
 
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer316 = Transformer(img_size=(input_h//16, input_w//16), patch_size=(2,2), in_channels=self.inplanes//2, hidden_size = self.inplanes*2, out_channels=self.inplanes, num_attention_heads = 16)   #res34


        
        ########  DECODER  ########

        self.cam1 = CAM(64,32)
        self.cam2 = CAM(64,32)

        self.dam_p3 = DAM_Position(128)
        self.dam_p4 = DAM_Position(256)
        self.dam_p5 = DAM_Position(512)


        self.dam_c3 = DAM_Channel(128)
        self.dam_c4 = DAM_Channel(256)
        self.dam_c5 = DAM_Channel(512)



        ###############

        self.deconv1 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
        self.deconv2 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
        self.deconv3 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
        self.deconv4 = self._make_transpose_layer(TransBottleneck, self.inplanes//2, 2, stride=2) # using // for python 3.6
        self.deconv5 = self._make_transpose_layer(TransBottleneck, n_class, 2, stride=2)
 
    def _make_transpose_layer(self, block, planes, blocks, stride=1):

        upsample = None
        if stride != 1:
            upsample = nn.Sequential(
                nn.ConvTranspose2d(self.inplanes, planes, kernel_size=2, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            ) 
        elif self.inplanes != planes:
            upsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride, padding=0, bias=False),
                nn.BatchNorm2d(planes),
            ) 
 
        for m in upsample.modules():
            if isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        layers = []

        for i in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes))

        layers.append(block(self.inplanes, planes, stride, upsample))
        self.inplanes = planes

        return nn.Sequential(*layers)
 
    def forward(self, rgb, disparity):


        verbose = False

        # encoder
        ######################################################################

        if verbose: print("rgb.size() original: ", rgb.size())  # (480, 640)
        if verbose: print("disparity.size() original: ", disparity.size()) # (480, 640)

        ######################################################################

        rgb = self.encoder_rgb_conv1(rgb)
        if verbose: print("rgb.size() after conv1: ", rgb.size()) # (240, 320)
        rgb = self.encoder_rgb_bn1(rgb)
        if verbose: print("rgb.size() after bn1: ", rgb.size())  # (240, 320)
        rgb = self.encoder_rgb_relu(rgb)
        if verbose: print("rgb.size() after relu: ", rgb.size())  # (240, 320)

        disparity = self.encoder_disparity_conv1(disparity)
        if verbose: print("disparity.size() after conv1: ", disparity.size()) # (240, 320)
        disparity = self.encoder_disparity_bn1(disparity)
        if verbose: print("disparity.size() after bn1: ", disparity.size()) # (240, 320)
        disparity = self.encoder_disparity_relu(disparity)
        if verbose: print("disparity.size() after relu: ", disparity.size())  # (240, 320)




        rgb = rgb + disparity
        rgb = self.cam1(rgb)

        if verbose: print("rgb.size() after fusion1: ", rgb.size())  # (240, 320)
        

        rgb = self.encoder_rgb_maxpool(rgb)
        if verbose: print("rgb.size() after maxpool: ", rgb.size()) # (120, 160)

        disparity = self.encoder_disparity_maxpool(disparity)
        if verbose: print("disparity.size() after maxpool: ", disparity.size()) # (120, 160)

        ######################################################################

        rgb = self.encoder_rgb_layer1(rgb)
        if verbose: print("rgb.size() after layer1: ", rgb.size()) # (120, 160)
        disparity = self.encoder_disparity_layer1(disparity)
        if verbose: print("disparity.size() after layer1: ", disparity.size()) # (120, 160)


        rgb = rgb + disparity
        rgb = self.cam2(rgb)

        if verbose: print("rgb.size() after fusion2: ", rgb.size())  # (240, 320)


        ######################################################################
 
        rgb = self.encoder_rgb_layer2(rgb)
        if verbose: print("rgb.size() after layer2: ", rgb.size()) # (60, 80)
        disparity = self.encoder_disparity_layer2(disparity)
        if verbose: print("disparity.size() after layer2: ", disparity.size()) # (60, 80)
     

        rgb = rgb + disparity
        rgb = self.dam_c3(rgb)+self.dam_p3(rgb)

        if verbose: print("rgb.size() after fusion3: ", rgb.size())  # (240, 320)
        ######################################################################

        rgb = self.encoder_rgb_layer3(rgb)
        if verbose: print("rgb.size() after layer3: ", rgb.size()) # (30, 40)
        disparity = self.encoder_disparity_layer3(disparity)
        if verbose: print("disparity.size() after layer3: ", disparity.size()) # (30, 40)
        

        rgb = rgb + disparity
        rgb = self.dam_c4(rgb)+self.dam_p4(rgb)
        

        if verbose: print("rgb.size() after fusion4: ", rgb.size())  # (240, 320)


        ######################################################################

        rgb = self.encoder_rgb_layer316(rgb)
        if verbose: print("rgb.size() after layer4: ", rgb.size()) # (15, 20)
        disparity = self.encoder_disparity_layer316(disparity)
        if verbose: print("disparity.size() after layer4: ", disparity.size()) # (15, 20)

        fuse = rgb + disparity
        fuse = self.dam_c5(fuse)+self.dam_p5(fuse)
  
        ######################################################################

        # decoder

        fuse = self.deconv1(fuse)
        if verbose: print("fuse after deconv1: ", fuse.size()) # (30, 40)
        fuse = self.deconv2(fuse)
        if verbose: print("fuse after deconv2: ", fuse.size()) # (60, 80)
        fuse = self.deconv3(fuse)
        if verbose: print("fuse after deconv3: ", fuse.size()) # (120, 160)
        fuse = self.deconv4(fuse)
        if verbose: print("fuse after deconv4: ", fuse.size()) # (240, 320)
        fuse = self.deconv5(fuse)
        if verbose: print("fuse after deconv5: ", fuse.size()) # (480, 640)

        return fuse
  
class TransBottleneck(nn.Module):
    # Ref from https://github.com/yuxiangsun/RTFNet
    def __init__(self, inplanes, planes, stride=1, upsample=None):
        super(TransBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)  
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)  
        self.bn2 = nn.BatchNorm2d(planes)

        if upsample is not None and stride != 1:
            self.conv3 = nn.ConvTranspose2d(planes, planes, kernel_size=2, stride=stride, padding=0, bias=False)  
        else:
            self.conv3 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)  

        self.bn3 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.upsample = upsample
        self.stride = stride
 
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.xavier_uniform_(m.weight.data)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            residual = self.upsample(x)

        out += residual
        out = self.relu(out)

        return out



class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, img_size, patch_size,in_channels,hidden_size):
        super(Embeddings, self).__init__()
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        self.patch_embeddings = nn.Conv2d(in_channels=in_channels,
                                       out_channels=hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, hidden_size))
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Attention(nn.Module):    
    def __init__(self,num_attention_heads,hidden_size):
        super(Attention, self).__init__()
        #self.vis = vis
        self.num_attention_heads = num_attention_heads
        self.hidden_size = hidden_size
        self.attention_head_size = int(self.hidden_size/ self.num_attention_heads)    #meigetoude chicun 
        self.all_head_size = self.num_attention_heads * self.attention_head_size      #suoyoutoude chicun ,yu shuru de xiangtong

        self.query = nn.Linear(self.hidden_size , self.all_head_size)
        self.key = nn.Linear(self.hidden_size,  self.all_head_size)
        self.value = nn.Linear(self.hidden_size , self.all_head_size)

        self.out = nn.Linear(self.hidden_size, self.hidden_size)
        self.attn_dropout = nn.Dropout(0.0)
        self.proj_dropout = nn.Dropout(0.0)

        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(16)

        self.softmax = nn.Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        #print("378:",attention_scores.size())
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = self.bn1(attention_scores)
        attention_probs = self.softmax(attention_scores)
        #weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = self.bn2(context_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class Mlp(nn.Module):
    def __init__(self,hidden_size):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size*4)   #mlp_dim = 1
        self.fc2 = nn.Linear(hidden_size*4, hidden_size)   #bianhuiqu
        self.act_fn = ACT2FN["gelu"]
        self.dropout = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    def __init__(self,hidden_size,num_attention_heads):
        super(Block, self).__init__()
        self.hidden_size = hidden_size
        self.attn = Attention(num_attention_heads,hidden_size)
        self.attention_norm = nn.LayerNorm(self.hidden_size , eps=1e-6)
        self.ffn_norm = nn.LayerNorm(self.hidden_size , eps=1e-6)
        self.Mlp = Mlp(self.hidden_size)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.Mlp(x)
        x = x + h
        return x

class Encoder(nn.Module):
    def __init__(self,hidden_size,num_attention_heads):
        super(Encoder, self).__init__()
        self.layer1 = Block(hidden_size,num_attention_heads)
        self.layer2 = Block(hidden_size,num_attention_heads)
        self.layer3 = Block(hidden_size,num_attention_heads)
        self.encoder_norm = nn.LayerNorm(hidden_size, eps=1e-6)

    def forward(self, hidden_states):
        hidden_states= self.layer1(hidden_states)
        hidden_states= self.layer2(hidden_states)
        hidden_states= self.layer3(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class Reshape(nn.Module):     # added by Zhen FENG
    def __init__(self,out_channels,hidden_size):
        super().__init__()
        #head_channels = 512
        self.conv_more = Conv2dReLU(
            hidden_size,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )

    def forward(self, hidden_states):
        B, n_patch, hidden = hidden_states.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch)) #need change
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)   # reshape
        x = self.conv_more(x)
        return x


class Transformer(nn.Module):
    def __init__(self, img_size, patch_size,in_channels, hidden_size,out_channels,num_attention_heads):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(img_size=img_size, patch_size = patch_size,in_channels = in_channels, hidden_size=hidden_size)
        self.encoder = Encoder(hidden_size,num_attention_heads)
        self.reshape_fz = Reshape(out_channels,hidden_size)

    def forward(self, input_ids):
        embedding_output= self.embeddings(input_ids)
        encoded= self.encoder(embedding_output)  # (B, n_patch, hidden)
        encoded = self.reshape_fz(encoded)
        return encoded


class CAM(nn.Module):     # # added by Zhen FENG
    def __init__(self, in_channels, med_channels):
        super(CAM, self).__init__()

        self.average = nn.AdaptiveAvgPool2d(1)

        self.fc1 = nn.Linear(in_channels,med_channels)
        self.bn1 = nn.BatchNorm1d(med_channels)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(med_channels,in_channels)
        self.sg = nn.Sigmoid()
    
    def forward(self,input):
        x = input
        x = self.average(input)
        x = x.squeeze(2)
        x = x.squeeze(2)
        x = self.fc1(x)
        x= self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sg(x)
        x = x.unsqueeze(2)
        x = x.unsqueeze(3)
        out = torch.mul(input,x)
        return out


class DAM_Position(nn.Module):
    """ Position attention submodule in Dual Attention Module"""
    # Ref from https://github.com/junfu1115/DANet
    def __init__(self, in_dim):
        super(DAM_Position, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x ([torch.tensor]): size N*C*H*W

        Returns:
            [torch.tensor]: size N*C*H*W
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class DAM_Channel(nn.Module):
    """ Channel attention submodule in Dual Attention Module """
    # Ref from https://github.com/junfu1115/DANet
    def __init__(self, in_dim):
        super(DAM_Channel, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)

    def forward(self,x):
        """
        Args:
            x ([torch.tensor]): size N*C*H*W

        Returns:
            [torch.tensor]: size N*C*H*W
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out

def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 512, 512).cuda(0)
    disparity = torch.randn(num_minibatch, 3, 512, 512).cuda(0)
    rtf = MAFNet(2).cuda(0)
    x = rtf(rgb,disparity)
    print('x: ', x.size())

if __name__ == '__main__':
    unit_test()
