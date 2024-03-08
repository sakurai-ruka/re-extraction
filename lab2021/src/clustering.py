"""
クラスタリング
"""

import os.path as osp
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import (dendrogram, fcluster, linkage,
                                     set_link_color_palette)

def main(args):
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Hiragino Maru Gothic Pro', 'Yu Gothic', 'Meirio', 'Takao', 'IPAexGothic', 'IPAPGothic', 'VL PGothic', 'Noto Sans CJK JP']
    method = args.method

    root_path = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
    inf_path = osp.join(root_path, f"outputs/extracted_features_{method}.df.pickle")
    out_img_path = osp.join(root_path, f"outputs/result_{method}.png")
    out_path = osp.join(root_path, f"linkage_{method}")

    df = pd.read_pickle(inf_path)

    set_link_color_palette(['purple', 'lawngreen', 'green', 'blue', 'orange', 'red'])
    z = linkage(df.iloc[:4000,5:], method='ward')
    np.save(out_path, z)
    # z = np.load(out_path + '.npy')

    fig = plt.figure(figsize=(20, 218), dpi=300)
    ax = fig.add_subplot(1, 1, 1, title="dendrogram")
    dendrogram(z, labels=df.iloc[:4000,4].values.tolist(),
                    leaf_font_size=5,
                    orientation='right',
                    #truncate_mode="level", p=10,
                    # no_plot=True,
                    distance_sort=True,
                    above_threshold_color='black')
    fig.subplots_adjust(left=0.2, right=1.1, top=1.0, bottom=0)
    fig.savefig(out_img_path)
    plt.show()

    exit()
    clusters = fcluster(z, 10, criterion='distance')
    result = []
    for i, c in enumerate(clusters):
        result.append([c, df.iloc[i,4]])


    aa = pd.DataFrame(result, columns=["cluster", "sentence"])
    aa = aa.sort_values('cluster').reset_index()
    print(aa)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str,
        help="feature's method", default='tfidf')
    parser.add_argument('--mecab_dict_path', type=str,
    help='Path to MeCab custom dictionary.',
    default='/usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')
    args = parser.parse_args()

    main(args)
