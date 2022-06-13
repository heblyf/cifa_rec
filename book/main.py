import os
import torch
import argparse
import numpy as np

from models import CIFARecModel
from trainers import CIFARecTrainer
from utils import (
    EarlyStopping,
    check_path,
    set_seed,
    get_local_time,
    # get_seq_dic,
    get_dataloder,
    get_rating_matrix,
)
from data.preprocess import get_seq_dic


def main():
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./book/data/", type=str)
    parser.add_argument("--output_dir", default="./book/output/", type=str)
    parser.add_argument("--data_name", default="book_recommend", type=str)
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument("--load_model", default=None, type=str)

    # model args
    parser.add_argument("--model_name", default="CIFARec", type=str)
    parser.add_argument("--hidden_size", default=16, type=int, help="hidden size of model")
    parser.add_argument(
        "--num_hidden_layers",
        default=1,
        type=int,
        help="number of filter-enhanced blocks",
    )
    parser.add_argument("--num_attention_heads", default=1, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str)  # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument(
        "--no_filters",
        action="store_true",
        help="if no filters, filter layers transform to self-attention",
    )

    parser.add_argument("--max_rating", default=10, type=int, help="maximum rating")
    parser.add_argument("--max_age", default=250, type=int, help="max age")

    # train args
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    parser.add_argument("--batch_size", default=256, type=int, help="number of batch_size")
    parser.add_argument("--epochs", default=1, type=int, help="number of epochs")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    parser.add_argument("--full_sort", action="store_true")
    parser.add_argument(
        "--patience",
        default=10,
        type=int,
        help="how long to wait after last time validation loss improved",
    )

    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument("--variance", default=5, type=float)

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    # 保存参数
    cur_time = get_local_time()
    if args.no_filters:
        args.model_name = "CIFARec"
    args_str = f"{args.model_name}-{args.data_name}"
    args.log_file = os.path.join(args.output_dir, args_str + ".txt")
    print(str(args))
    with open(args.log_file, "a") as f:
        f.write(str(args) + "\n")

    # 模型 checkpoint 地址
    args.checkpoint_path = os.path.join(args.output_dir, args_str + ".pt")

    # 加载数据
    # seq_dic   {'user_seq':[], 'num_users':[], 'sample_seq':[]}
    # max_item  item 的最大个数
    # num_users user 的最大个数
    seq_dic, max_item, num_users = get_seq_dic(args)
    # 使用 0 pad 序列，需预留 0
    args.item_size = max_item + 1
    args.max_rating = args.max_rating + 1
    args.num_users = num_users + 1

    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args, seq_dic)

    # 需要注意的是，序列 train 后两位没取，valid 取到了倒数第二位，test 取到了倒数第一位
    # 将用户过去的这一系列交互行为视作用户行为序列，并通过构建模型对其建模，来预测下一时刻用户最感兴趣的内容，这就是序列化推荐（Sequential Recommendation）的核心思想

    # 初始化模型
    model = CIFARecModel(args=args)

    # 初始化训练器
    trainer = CIFARecTrainer(model, train_dataloader, eval_dataloader, test_dataloader, args)

    if args.full_sort:
        # XXX full_sort 为了保证顺序
        args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)

    if args.do_eval:
        if args.load_model is None:
            print(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + ".pt")
            trainer.load(args.checkpoint_path)
            print(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0, full_sort=args.full_sort)

    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            scores, _ = trainer.valid(epoch, full_sort=args.full_sort)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("---------------Sample 99 results---------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path))
        scores, result_info = trainer.test(0, full_sort=args.full_sort)

    print(args_str)
    print(result_info)
    with open(args.log_file, "a") as f:
        f.write(args_str + "\n")
        f.write(result_info + "\n")


if __name__ == "__main__":
    main()
