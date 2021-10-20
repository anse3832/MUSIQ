import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class IQARegression(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # channel conversion for encoder input
        self.conv_enc = nn.Conv2d(in_channels=2048, out_channels=config.d_hidn, kernel_size=1)

        # transformer (only encoder)
        self.transformer = Transformer(self.config)

        # final MLP head
        self.projection = nn.Sequential(
            nn.Linear(self.config.d_hidn, self.config.d_MLP_head, bias=False),
            nn.GELU(),
            nn.Linear(self.config.d_MLP_head, self.config.n_output, bias=False)
        )
    
    
    def forward(self, mask_inputs, feat_dis_org, feat_dis_scale_1, feat_dis_scale_2):
        # batch x (C=2048) x H x W -> batch x (C=384) x H x W
        feat_dis_org_embed = self.conv_enc(feat_dis_org)
        feat_dis_scale_1_embed = self.conv_enc(feat_dis_scale_1)
        feat_dis_scale_2_embed = self.conv_enc(feat_dis_scale_2)
        
        # batch x (C=384) x H x W -> batch x (H*W) x (C=384)
        # b, c, h, w = enc_inputs_embed.size()
        # enc_inputs_embed = torch.reshape(enc_inputs_embed, (b, c, h*w))
        # enc_inputs_embed = enc_inputs_embed.permute(0, 2, 1)

         # (bs, n_enc_seq+1, d_hidn) -> (bs, d_hidn)
        enc_outputs = self.transformer(mask_inputs, feat_dis_org_embed, feat_dis_scale_1_embed, feat_dis_scale_2_embed)
        enc_outputs = enc_outputs[:, 0, :]  # cls token

        # (bs, n_output)
        pred = self.projection(enc_outputs)
        
        return pred


""" transformer """
class Transformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = Encoder(self.config)
    
    def forward(self, mask_inputs, feat_dis_org_embed, feat_dis_scale_1_embed, feat_dis_scale_2_embed):

        # enc_inputs: batch x n_enc_seq+1
        # feat_dis_org_embed: batch x (C=384) x H x W

        # (bs, n_enc_seq+1, d_hidn), [(bs, n_head, n_enc_seq+1, n_enc_seq+1)]
        enc_outputs, enc_self_attn_probs = self.encoder(mask_inputs, feat_dis_org_embed, feat_dis_scale_1_embed, feat_dis_scale_2_embed)
        
        return enc_outputs


""" encoder """
class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # learnable scale embedding 
        self.scale_org_embedding = nn.Parameter(torch.rand(1, self.config.d_hidn, 1, 1))
        self.scale_1_embedding = nn.Parameter(torch.rand(1, self.config.d_hidn, 1, 1))
        self.scale_2_embedding = nn.Parameter(torch.randn(1, self.config.d_hidn, 1, 1))

        # learnable 2D spatial embedding
        self.pos_embedding = nn.Parameter(torch.randn(1, self.config.Grid, self.config.Grid, self.config.d_hidn))

        # cls token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.config.d_hidn))
        self.dropout = nn.Dropout(self.config.emb_dropout)

        self.layers = nn.ModuleList([EncoderLayer(self.config) for _ in range(self.config.n_layer)])
    
    def forward(self, mask_inputs, feat_dis_org_embed, feat_dis_scale_1_embed, feat_dis_scale_2_embed):
        # feat_dis_org_embed: batch x (C=384) x (H=24) x (W=32)
        # feat_dis_scale_1_embed: batch x (C=384) x (H=9) x (W=12)
        # feat_dis_scale_2_embed: batch x (C=384) x (H=5) x (W=7)
          
        # learnable scale embedding
        scale_org_embed = repeat(self.scale_org_embedding, '() c () () -> b c h w', b=self.config.batch_size, h=24, w=32)
        scale_1_embed = repeat(self.scale_1_embedding, '() c () () -> b c h w', b=self.config.batch_size, h=9, w=12)
        scale_2_embed = repeat(self.scale_1_embedding, '() c () () -> b c h w', b=self.config.batch_size, h=5, w=7)

        feat_dis_org_embed += scale_org_embed
        feat_dis_scale_1_embed += scale_1_embed
        feat_dis_scale_2_embed += scale_2_embed

        # learnable 2D spatial embedding
        # original scale
        b, c, h, w = feat_dis_org_embed.size()
        spatial_org_embed = torch.zeros(1, self.config.d_hidn, h, w).to(self.config.device)
        for i in range(h):
            for j in range(w):
                t_i = int((i/h)*self.config.Grid)
                t_j = int((j/w)*self.config.Grid)
                spatial_org_embed[:, :, i, j] = self.pos_embedding[:, t_i, t_j, :]
        spatial_org_embed = repeat(spatial_org_embed, '() c h w -> b c h w', b=self.config.batch_size)
        # scale 1
        b, c, h, w = feat_dis_scale_1_embed.size()
        spatial_scale_1_embed = torch.zeros(1, self.config.d_hidn, h, w).to(self.config.device)
        for i in range(h):
            for j in range(w):
                t_i = int((i/h)*self.config.Grid)
                t_j = int((j/w)*self.config.Grid)
                spatial_scale_1_embed[:, :, i, j] = self.pos_embedding[:, t_i, t_j, :]
        spatial_scale_1_embed = repeat(spatial_scale_1_embed, '() c h w -> b c h w', b=self.config.batch_size)
        # scale 2
        b, c, h, w = feat_dis_scale_2_embed.size()
        spatial_scale_2_embed = torch.zeros(1, self.config.d_hidn, h , w).to(self.config.device)
        for i in range(h):
            for j in range(w):
                t_i = int((i/h)*self.config.Grid)
                t_j = int((j/w)*self.config.Grid)
                spatial_scale_2_embed[:, :, i, j] = self.pos_embedding[:, t_i, t_j, :]
        spatial_scale_2_embed = repeat(spatial_scale_2_embed, '() c h w -> b c h w', b=self.config.batch_size)

        feat_dis_org_embed += spatial_org_embed
        feat_dis_scale_1_embed += spatial_scale_1_embed
        feat_dis_scale_2_embed += spatial_scale_2_embed

        # batch x (C=384) x (H=24) x (W=32) -> batch x (H*W=24*32) x (C=384)
        b, c, h, w = feat_dis_org_embed.size()
        feat_dis_org_embed = torch.reshape(feat_dis_org_embed, (b, c, h*w))
        feat_dis_org_embed = feat_dis_org_embed.permute((0, 2, 1))
        # batch x (C=384) x (H=12) x (W=9) -> batch x (H*W=12*9) x (C=384)
        b, c, h, w = feat_dis_scale_1_embed.size()
        feat_dis_scale_1_embed = torch.reshape(feat_dis_scale_1_embed, (b, c, h*w))
        feat_dis_scale_1_embed = feat_dis_scale_1_embed.permute((0, 2, 1))
        # batch x (C=384) x (H=7) x (W=5) -> batch x (H*W=7*5) x (C=384)
        b, c, h, w = feat_dis_scale_2_embed.size()
        feat_dis_scale_2_embed = torch.reshape(feat_dis_scale_2_embed, (b, c, h*w))
        feat_dis_scale_2_embed = feat_dis_scale_2_embed.permute((0, 2, 1))
        # concat scale embedding
        inputs_embed = torch.cat((feat_dis_org_embed, feat_dis_scale_1_embed, feat_dis_scale_2_embed), dim=1)

        # outputs: batch x (len_seq+1) x n_feat
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=self.config.batch_size)
        x = torch.cat((cls_tokens, inputs_embed), dim=1)
        
        # x += self.pos_embedding     # positional embedding (learnable parameter)
        outputs = self.dropout(x)        

        # (bs, n_enc_seq+1, n_enc_seq+1)
        attn_mask = get_attn_pad_mask(mask_inputs, mask_inputs, self.config.i_pad)

        attn_probs = []
        for layer in self.layers:
            # (bs, n_enc_seq+1, d_hidn), (bs, n_head, n_enc_seq+1, n_enc_seq+1)
            outputs, attn_prob = layer(outputs, attn_mask)
            attn_probs.append(attn_prob)
        
        # (bs, n_enc_seq+1, d_hidn), [(bs, n_head, n_enc_seq+1, n_enc_seq+1)]
        return outputs, attn_probs


""" encoder layer """
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.self_attn = MultiHeadAttention(self.config)
        self.layer_norm1 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
        self.pos_ffn = PoswiseFeedForwardNet(self.config)
        self.layer_norm2 = nn.LayerNorm(self.config.d_hidn, eps=self.config.layer_norm_epsilon)
    
    def forward(self, inputs, attn_mask):
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        att_outputs, attn_prob = self.self_attn(inputs, inputs, inputs, attn_mask)
        att_outputs = self.layer_norm1(inputs + att_outputs)
        
        # (bs, n_enc_seq, d_hidn)
        ffn_outputs = self.pos_ffn(att_outputs)
        ffn_outputs = self.layer_norm2(ffn_outputs + att_outputs)
        
        # (bs, n_enc_seq, d_hidn), (bs, n_head, n_enc_seq, n_enc_seq)
        return ffn_outputs, attn_prob


def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table


""" attention pad mask """
def get_attn_pad_mask(seq_q, seq_k, i_pad):
    batch_size, len_q = seq_q.size()
    batch_size, len_k = seq_k.size()
    pad_attn_mask = seq_k.data.eq(i_pad)
    pad_attn_mask= pad_attn_mask.unsqueeze(1).expand(batch_size, len_q, len_k)
    return pad_attn_mask


""" multi head attention """
class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.W_Q = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_K = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.W_V = nn.Linear(self.config.d_hidn, self.config.n_head * self.config.d_head)
        self.scaled_dot_attn = ScaledDotProductAttention(self.config)
        self.linear = nn.Linear(self.config.n_head * self.config.d_head, self.config.d_hidn)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, Q, K, V, attn_mask):
        batch_size = Q.size(0)
        
        # (bs, n_head, n_q_seq, d_head)
        q_s = self.W_Q(Q).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        # (bs, n_head, n_k_seq, d_head)
        k_s = self.W_K(K).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)
        # (bs, n_head, n_v_seq, d_head)
        v_s = self.W_V(V).view(batch_size, -1, self.config.n_head, self.config.d_head).transpose(1,2)

        # (bs, n_head, n_q_seq, n_k_seq)
        attn_mask = attn_mask.unsqueeze(1).repeat(1, self.config.n_head, 1, 1)

        # (bs, n_head, n_q_seq, d_head), (bs, n_head, n_q_seq, n_k_seq)
        context, attn_prob = self.scaled_dot_attn(q_s, k_s, v_s, attn_mask)
        # (bs, n_head, n_q_seq, h_head * d_head)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.config.n_head * self.config.d_head)
        # (bs, n_head, n_q_seq, e_embd)
        output = self.linear(context)
        output = self.dropout(output)
        
        # (bs, n_q_seq, d_hidn), (bs, n_head, n_q_seq, n_k_seq)
        return output, attn_prob


""" scale dot product attention """
class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dropout = nn.Dropout(config.dropout)
        self.scale = 1 / (self.config.d_head ** 0.5)
    
    def forward(self, Q, K, V, attn_mask):
        # (bs, n_head, n_q_seq, n_k_seq)
        scores = torch.matmul(Q, K.transpose(-1, -2))
        scores = scores.mul_(self.scale)
        scores.masked_fill_(attn_mask, -1e9)
        # (bs, n_head, n_q_seq, n_k_seq)
        attn_prob = nn.Softmax(dim=-1)(scores)
        attn_prob = self.dropout(attn_prob)
        # (bs, n_head, n_q_seq, d_v)
        context = torch.matmul(attn_prob, V)
        
        # (bs, n_head, n_q_seq, d_v), (bs, n_head, n_q_seq, n_v_seq)
        return context, attn_prob


""" feed forward """
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.conv1 = nn.Conv1d(in_channels=self.config.d_hidn, out_channels=self.config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=self.config.d_ff, out_channels=self.config.d_hidn, kernel_size=1)
        self.active = F.gelu
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, inputs):
        # (bs, d_ff, n_seq)
        output = self.conv1(inputs.transpose(1, 2))
        output = self.active(output)
        # (bs, n_seq, d_hidn)
        output = self.conv2(output).transpose(1, 2)
        output = self.dropout(output)
        
        # (bs, n_seq, d_hidn)
        return output