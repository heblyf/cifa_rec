import os
import argparse
import random
import pandas as pd
import numpy as np
import pickle


def user_data_clean_up(users):
    # 用平均值填充年龄
    fix_users_age = users[users.Age > 0]
    fix_users_age = users[users.Age < 100]
    users["Age"] = users["Age"].fillna(int(fix_users_age["Age"].mean()))
    # users = users.loc[users["Age"] < 100]

    # 处理地区
    location = users.Location.str.split(", ", expand=True)
    users["City"] = location[0].str.title()
    users["State"] = location[1].str.title()
    users["Country"] = location[2].str.title()

    return users


def load_data(args):
    users = pd.read_csv(
        os.path.join(args.data_dir, "users.csv"),
        sep=";",
        encoding="CP1252",
        escapechar="\\",
    )
    users = user_data_clean_up(users)

    books = pd.read_csv(
        os.path.join(args.data_dir, "books.csv"),
        sep=";",
        encoding="CP1252",
        escapechar="\\",
    )
    books["Book-ID"] = books.index + 1  # book id

    ratings = pd.read_csv(
        os.path.join(args.data_dir, "ratings.csv"),
        sep=";",
        encoding="CP1252",
        escapechar="\\",
    )
    # 将 ISBN 替换为 Book-ID
    ratings["Book-ID"] = ratings["ISBN"].map(books.set_index("ISBN")["Book-ID"])
    ratings = ratings.dropna(subset=["Book-ID"])

    return users, books, ratings


def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)  # 这个是随机生成的
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item


def neg_sample_items(item_set, item_size):
    neg_samples = []
    item = random.randint(1, item_size - 1)
    while len(neg_samples) <= 50:
        item = random.randint(1, item_size - 1)
        if item not in item_set:
            neg_samples.append(item)
            item_set = item_set | set([item])  # 相当于item_set和set([item])合并了
    return neg_samples


def gen_seq_dic(args):
    """
    生成 seq dict
    user_id_seq     所有用户的 id
    user_seq        用户的行为序列 即每一个用户评价过哪些电影的 id
    rating_seq      用户行为序列对每一个 item 的评分
    sample_seq      负采样的 itemid
    num_users       用户数量
    max_item        最大 item 的 id
    max_rating      最大的评分
    """
    users, books, ratings = load_data(args)

    user_ids, num_users = users["User-ID"], len(users)

    user_age = list(users["Age"])
    max_age = max(user_age)

    max_item = max(books["Book-ID"])
    args.item_size = max_item + 1

    max_rating = max(ratings["Book-Rating"])

    user_seq = []
    user_age_seq = []
    sample_seq = []
    rating_seq = []
    user_id_seq = []
    count = 0
    for i in range(args.limit):
        user_id = user_ids[i]
        # 用户交互的 book 序列
        book_seq = ratings.loc[ratings["User-ID"] == user_id]
        if len(book_seq) == 0:
            continue

        user_id_seq.append(user_id)

        # 用户年龄
        user_age_seq.append(user_age[i])

        book_id_seq = list(book_seq["Book-ID"])
        user_seq.append(book_id_seq)

        # rating 序列
        user_rating_seq = list(book_seq["Book-Rating"])
        rating_seq.append(user_rating_seq)

        # 生成负采样序列
        seq_set = set(book_id_seq)
        user_sample_seq = neg_sample_items(seq_set, args.item_size)
        sample_seq.append(user_sample_seq)

        count += 1
        if count % 100 == 0:
            print(count)

    seq_dic = {
        "user_id_seq": user_id_seq,
        "user_age_seq": user_age_seq,
        "user_seq": user_seq,
        "rating_seq": rating_seq,
        "sample_seq": sample_seq,
        "num_users": num_users,
        "max_age": max_age,
        "max_item": max_item,
        "max_rating": max_rating,
    }

    seq_dic_path = os.path.join(args.data_dir, "seq_dict.pkl")
    seq_dic_file = open(seq_dic_path, "wb")
    pickle.dump(seq_dic, seq_dic_file)
    seq_dic_file.close()


def get_seq_dic(args):
    seq_dic_path = os.path.join(args.data_dir, "seq_dict.pkl")
    if not os.path.exists(seq_dic_path):
        gen_seq_dic(args)

    seq_dic_file = open(seq_dic_path, "rb")
    seq_dic = pickle.load(seq_dic_file)
    seq_dic_file.close()

    return seq_dic, seq_dic["max_item"], seq_dic["num_users"]


if __name__ == "__main__":
    # 参数设置
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="book/data", type=str)
    parser.add_argument("--limit", default=10000, type=int)

    args = parser.parse_args()

    gen_seq_dic(args=args)

    seq_dic, max_item, num_users = get_seq_dic(args)

    print(seq_dic["user_id_seq"][:5])

    print(seq_dic["user_seq"][:5])

    print(seq_dic["rating_seq"][:5])

    print(seq_dic["max_rating"])
    print(seq_dic["max_item"])
    print(seq_dic["num_users"])
    print(seq_dic["max_age"])
