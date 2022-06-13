import torch
import torch.nn as nn
from modules import Encoder, LayerNorm


class FMLPRecModel(nn.Module):
    def __init__(self, args):
        super(FMLPRecModel, self).__init__()
        self.args = args
        # item embedding
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        # position embedding
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        # TODO rating 作为一种特征
        # TODO item 属性，用 Genres
        # 归一化层
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        # dropout
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        # 编码器
        # self attention 或者 fft
        self.item_encoder = Encoder(args)

        # pytorch 中的 model.apply(fn) 会递归地将函数 fn 应用到父模块的每个子模块 submodule，也包括 model 自身
        self.apply(self.init_weights)
        print("FMLPRecModel init succeed.")

    def add_position_embedding(self, sequence):
        seq_length = sequence.size(1)  # [256,50]
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)  # 位置信息 [50]
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)  # [256,50]
        item_embeddings = self.item_embeddings(sequence)  # 项目的embeddings
        position_embeddings = self.position_embeddings(position_ids)
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # same as SASRec
    def forward(self, input_ids):
        # input_ids batch_size*max_seq_length 输入都是 padded
        attention_mask = (input_ids > 0).long()  # 大于0的变成1 否则是0  [256,50]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)  # torch.int64  ([256,1,1,50])
        max_len = attention_mask.size(-1)  # 50
        attn_shape = (1, max_len, max_len)  # [1,50,50]
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1)  # torch.uint8   返回矩阵上三角部分，其余部分定义为0
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()  # ([1,1,50,50])

        if self.args.cuda_condition:
            subsequent_mask = subsequent_mask.cuda()
        extended_attention_mask = extended_attention_mask * subsequent_mask  # ([256,1,50,50])
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(
            sequence_emb,
            extended_attention_mask,
            output_all_encoded_layers=True,
        )
        sequence_output = item_encoded_layers[-1]

        return sequence_output

    def init_weights(self, module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        elif isinstance(module, LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
