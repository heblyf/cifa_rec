import os
import argparse
import random
import pandas as pd
import numpy as np
import pickle


def genres_multi_hot(genre_int_map):
    def helper(genres):
        genre_int_list = [genre_int_map[genre] for genre in genres.split("|")]
        multi_hot = np.zeros(len(genre_int_map), dtype=int)
        multi_hot[genre_int_list] = 1
        # return "|".join(list(multi_hot))
        return list(multi_hot)

    return helper


def load_data(args):
    """加载数据并处理"""
    users_data_path = os.path.join(args.data_dir, "users.dat")
    movies_data_path = os.path.join(args.data_dir, "movies.dat")
    ratings_data_path = os.path.join(args.data_dir, "ratings.dat")

    # 加载用户信息表
    users_fields = ["UserID", "Gender", "Age", "JobID", "Zip-code"]
    users = pd.read_table(users_data_path, sep="::", header=None, names=users_fields, engine="python")

    # 映射用户性别
    gender_map = {"F": 0, "M": 1}
    users["GenderIndex"] = users["Gender"].map(gender_map)

    # 映射用户年龄
    age_map = {val: ii for ii, val in enumerate(set(users["Age"]))}
    age_map = {1: 0, 18: 1, 25: 2, 35: 3, 45: 4, 50: 5, 56: 6}
    users["AgeIndex"] = users["Age"].map(age_map)

    # 加载电影信息表
    movies_title = ["MovieID", "Title", "Genres"]
    movies = pd.read_table(
        movies_data_path,
        sep="::",
        header=None,
        names=movies_title,
        engine="python",
    )

    # 电影题材 Multi-Hot 编码
    genre_set = set()
    for val in movies["Genres"].str.split("|"):
        genre_set.update(val)
    genre_int_map = {val: ii for ii, val in enumerate(genre_set)}
    movies["GenresMultiHot"] = movies["Genres"].map(genres_multi_hot(genre_int_map))

    # 加载 rating 表
    ratings_title = ["UserID", "MovieID", "ratings", "timestamps"]
    ratings = pd.read_table(
        ratings_data_path,
        sep="::",
        header=None,
        names=ratings_title,
        engine="python",
    )

    return users, movies, ratings


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
    user_gender_seq 用户的性别序列
    user_age_seq    用户的年龄序列
    user_seq        用户的行为序列 即每一个用户评价过哪些电影的 id，已经按照时间戳进行排序
    rating_seq      用户行为序列对每一个 item 的评分
    genres_seq      电影题材的序列 已经做了 Multi-Hot 编码
    sample_seq      负采样的 itemid
    num_users       用户数量
    max_item        最大 item 的 id
    max_rating      最大的评分
    """
    users, movies, ratings = load_data(args)
    user_ids, num_users = users["UserID"], len(users)
    max_item = max(movies["MovieID"])
    args.item_size = max_item + 1

    user_id_seq = list(user_ids)
    user_gender_seq = list(users["GenderIndex"])
    user_age_seq = list(users["AgeIndex"])

    user_seq = []
    sample_seq = []
    rating_seq = []
    genres_seq = []
    count = 0
    for user_id in user_ids:
        # user 评分 movie 的序列
        movie_seq = ratings.loc[ratings["UserID"] == user_id].sort_values("timestamps")
        movie_id_seq = list(movie_seq["MovieID"])
        user_seq.append(movie_id_seq)

        # rating
        user_rating_seq = list(movie_seq["ratings"])
        rating_seq.append(user_rating_seq)

        genres = []
        for movie_id in movie_id_seq:
            movie = movies.loc[movies["MovieID"] == movie_id]
            genres.append(movie["GenresMultiHot"].values[0])

        genres_seq.append(genres)

        # 生成负采样序列
        seq_set = set(movie_id_seq)
        user_sample_seq = neg_sample_items(seq_set, args.item_size)
        sample_seq.append(user_sample_seq)

        count += 1
        if count % 100 == 0:
            print(count)

    seq_dic = {
        "user_id_seq": user_id_seq,
        "user_gender_seq": user_gender_seq,
        "user_age_seq": user_age_seq,
        "user_seq": user_seq,
        "rating_seq": rating_seq,
        "genres_seq": genres_seq,
        "sample_seq": sample_seq,
        "num_users": num_users,
        "max_item": max_item,
        "max_rating": 5,
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
    parser.add_argument("--data_dir", default="", type=str)

    args = parser.parse_args()

    gen_seq_dic(args=args)

    seq_dic, max_item, num_users = get_seq_dic(args)

    print(seq_dic["user_id_seq"][0])
    print(seq_dic["user_gender_seq"][0])
    print(seq_dic["user_age_seq"][0])

    print(len(seq_dic["user_seq"][0]))
    print(len(seq_dic["rating_seq"][0]))
    print(len(seq_dic["genres_seq"][0]))
    print(max_item)
