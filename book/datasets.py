import torch
import random
from torch.utils.data import Dataset


class CIFARecDataset(Dataset):
    """
    
    """
    def __init__(self, args, seq_dic, test_neg_items=None, data_type="train"):
        self.args = args
        self.user_seq = []
        self.user_age_seq = []
        self.max_len = args.max_seq_length

        self.rating_seq = []

        self.user_id_seq = []

        user_seq = seq_dic["user_seq"]
        user_age_seq = seq_dic["user_age_seq"]
        user_id_seq = seq_dic["user_id_seq"]
        rating_seq = seq_dic["rating_seq"]

        # 训练集：取 1~n-2，并且构造如下样本 1、1~2...1~n-2
        # 验证集：取 1~n-1
        # 测试集：取 1~n
        if data_type == "train":
            n_total = len(user_id_seq)
            for j in range(n_total):
                c = 0
                input_ids = user_seq[j][-(self.max_len + 2) : -2]  # keeping same as train set
                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[: i + 1])
                    c += 1
                ratings = rating_seq[j][-(self.max_len + 2) : -2]
                for i in range(len(ratings)):
                    self.rating_seq.append(ratings[: i + 1])
                for i in range(c):
                    self.user_id_seq.append(user_id_seq[j])
                    self.user_age_seq.append(user_age_seq[j])
        elif data_type == "valid":
            n_total = len(user_id_seq)
            for i in range(n_total):
                # 验证阶段，需序列长度大于 1
                if len(user_seq[i]) == 1:
                    continue
                self.user_seq.append(user_seq[i][:-1])
                self.rating_seq.append(rating_seq[i][:-1])
                self.user_id_seq.append(user_id_seq[i])
                self.user_age_seq.append(user_age_seq[i])
        else:
            self.user_seq = user_seq
            self.rating_seq = rating_seq
            self.user_id_seq = user_id_seq
            self.user_age_seq = user_age_seq

        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __len__(self):
        return len(self.user_seq)

    def __getitem__(self, index):
        # 每个输入序列，用前 n-1 序号预测第 n 个，即 input_ids 预测 answer
        items = self.user_seq[index]
        input_ids = items[:-1]
        answer = items[-1]

        ratings = self.rating_seq[index][:-1]

        user_id = self.user_id_seq[index]
        user_age = self.user_age_seq[index]

        seq_set = set(items)

        # 获取负 answer，和输入 items 不同即可
        neg_answer = neg_sample(seq_set, self.args.item_size)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len :]
        assert len(input_ids) == self.max_len

        ratings = [0] * pad_len + ratings
        ratings = ratings[-self.max_len :]
        assert len(ratings) == self.max_len

        if self.test_neg_items is not None:
            # XXX
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(ratings, dtype=torch.long),
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(user_age, dtype=torch.long),
                # torch.tensor(attribute, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(ratings, dtype=torch.long),
                torch.tensor(user_id, dtype=torch.long),
                torch.tensor(user_age, dtype=torch.long),
                # torch.tensor(attribute, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
            )

        return cur_tensors


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)  # 这个是随机生成的
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item
