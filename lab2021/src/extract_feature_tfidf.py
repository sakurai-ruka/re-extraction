"""
特徴抽出
"""
import argparse
import os.path as osp

import MeCab
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from os import environ
import matplotlib.pyplot as plt

tagger = ""

def write_idf_rank(feature_names: np.ndarray, idf: np.ndarray, index: np.ndarray, outf: str) -> None:
    result = ""
    for i, (text, idf__) in enumerate(zip(feature_names[index], idf[index])):
        result += f'{i}, {idf__:.4f}, {text}\n'

    root_path = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
    out_path = osp.join(root_path, "outputs", outf)
    with open(out_path, "w") as f:
        f.write(result)

def plot_idf_ratio_graph(idf: np.ndarray, index: np.ndarray, outf: str) -> None:
    fig = plt.figure(figsize=(20, 20), dpi=300)
    ax = fig.add_subplot(1, 1, 1, title="Idf ratio graph")
    x = np.arange(len(index))
    plt.plot(x, idf[index])

    root_path = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
    out_path = osp.join(root_path, "outputs", outf)
    fig.savefig(out_path)

def main(args):
    global tagger

    # Mecab Tagger の準備
    tagger = MeCab.Tagger('-d {} -Ochasen'.format(args.mecab_dict_path)) if args.mecab_dict_path is not None else MeCab.Tagger('-Ochasen')
    vectorizer = TfidfVectorizer(token_pattern=u'(?u)\\b\\w+\\b', max_df=1.0, max_features=None, binary=True, use_idf=True)

    root_path = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
    inf_path = osp.join(root_path, 'outputs/data.csv')

    df = pd.read_csv(inf_path)
    sentences = df['sentence'].values


    type_ = "surface" if args.is_surface else "origin"
    docs = [mecab_processing(sentence, type_) for sentence in sentences]
    vecs = vectorizer.fit_transform(docs)

    feature_names = np.asarray(vectorizer.get_feature_names())
    idf = vectorizer.idf_
    idf_vec = idf * -1
    index = idf_vec.argsort()
    list_ = feature_names[index]

    # ダンプ
    # write_idf_rank(feature_names, idf, index, f"idf_rank_{type_}.txt")
    # plot_idf_ratio_graph(idf, index, f"idf_ratio_{type_}.png")

    feature_array = vecs.toarray()
    num_feature_dims = feature_array.shape[1]
    columns = ['x{}'.format(i) for i in range(num_feature_dims)]
    feature_df = pd.DataFrame(feature_array, columns=columns)
    df_all = pd.concat([df, feature_df], axis=1)

    #out_csv_path = osp.join(root_path, f"outputs/extracted_features_{args.method}_{type_}.csv")
    out_pkl_path = osp.join(root_path, f"outputs/extracted_features_{args.method}_{type_}.df.pickle")
    # df_all.to_csv(out_csv_path, index=None)
    df_all.to_pickle(out_pkl_path)

def preprocessing(sentence):
    return sentence.rstrip()

def mecab_processing(sentence: str, type_: str = "surface", part: str = ""):
    # chasenの出力形式に合わせたindex
    dic_ = {"origin": 2,
            "surface": 0}
    sentence = preprocessing(sentence)
    parsed = tagger.parse(sentence)
    parsed = parsed.split("EOS")[0]
    list_word = [word_info.split("\t")[dic_[type_]] for word_info in parsed.split("\n")[:-1] \
            if part in word_info.split("\t")[3]]
    return " ".join(list_word)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str,
        help="feature's method", default='tfidf')
    parser.add_argument('-s', '--is_surface', action="store_true",
        help="If want to use surface, specify this")
    parser.add_argument('-d', '--mecab_dict_path', type=str,
    help='Path to MeCab custom dictionary.',
    # default='/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
    default='/home/mtakahashi/miniconda3/envs/dev/lib/mecab/dic/mecab-ipadic-neologd')
    args = parser.parse_args()

    main(args)