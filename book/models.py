import torch
import torch.nn as nn
from modules import Encoder, AdaNorm


class FMLPRecModel(nn.Module):
    def __init__(self, args):
        super(FMLPRecModel, self).__init__()
        self.args = args
        # item embedding
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        # position embedding
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        # rating embedding
        self.rating_embeddings = nn.Embedding(args.max_rating, args.hidden_size)
        # user embedding
        self.user_embeddings = nn.Embedding(args.num_users, args.hidden_size)
        # user age
        self.user_age_embeddings = nn.Embedding(args.max_age, args.hidden_size)
        # AdaNorm layer
        self.AdaNorm = AdaNorm(args.hidden_size, eps=1e-12)
        # dropout
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        # 编码器
        # self attention 或者 fft
        self.item_encoder = Encoder(args)

        # pytorch 中的 model.apply(fn) 会递归地将函数 fn 应用到父模块的每个子模块 submodule，也包括 model 自身
        self.apply(self.init_weights)
        print("FMLPRecModel init succeed.")

    def embedding(self, sequence, ratings, users, users_age):
        # sequence batch_size*max_seq
        # ratings batch_size*max_seq
        # genres batch_size*max_seq*num_genres
        # users batch_size*1
        seq_length = sequence.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        # item
        item_embeddings = self.item_embeddings(sequence)
        # 位置信息
        position_embeddings = self.position_embeddings(position_ids)
        # rating
        rating_embeddings = self.rating_embeddings(ratings)

        batch_size = users.shape[0]
        users = users.unsqueeze(1).expand(batch_size, seq_length)
        users_age = users_age.unsqueeze(1).expand(batch_size, seq_length)
        # users
        user_embeddings = self.user_embeddings(users)
        # user age
        user_age_embeddings = self.user_age_embeddings(users_age)

        # XXX 目前的策略是把所有的 embedding 累加，再传入 encoder 层
        sequence_emb = item_embeddings + position_embeddings + rating_embeddings + user_embeddings + user_age_embeddings
        sequence_emb = self.AdaNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # same as SASRec
    def forward(self, input_ids, ratings, users, users_age):
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

        sequence_emb = self.embedding(input_ids, ratings, users, users_age)

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
        elif isinstance(module, AdaNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
