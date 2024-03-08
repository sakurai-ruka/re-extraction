"""
文のペアと推移関係ラベルを付与する
"""

import itertools
import os
from glob import glob


def generate_pos_data(fpath: str, mode: str) -> int:
    """anotateファイルから学習データを生成

    Args:
        fpath (str): anotateファイルのパス
        mode (str): pos or neg

    Returns:
        int: 1ファイルから作られた学習データのlen
    """
    fname = os.path.basename(fpath)

    lines = []
    with open(fpath, "r") as f:
        for line in f:
            if line != '\n':
                line = line.strip()
            lines.append(line)

    # 同様な意味の文を同一階層に格納する2階list
    # grouping_titles: [[文1,2], [3,4,5], [6]]
    grouping_titles = []
    group = []
    for line in lines:
        if line != "\n":
            group.append(line)
        else:
            grouping_titles.append(group)
            group = []
    grouping_titles.append(group)

    all_title_pair = itertools.combinations(itertools.chain.from_iterable(grouping_titles), 2)
    pair_within_group = list(itertools.chain.from_iterable([itertools.combinations(x,2) for x in grouping_titles]))

    result = [pair for pair in all_title_pair if pair not in pair_within_group]

    out_s = "x1,x2,label\n"
    for pair in result:
        out_s += ",".join(pair) + ',0\n'

    out_path = "/home/mtakahashi/work/lab2021/outputs/0109_transition_pairs/neg_data"
    with open(os.path.join(out_path, fname), "w") as f:
        f.write(out_s)
    return len(result)



mode_list = ['pos', 'neg']
for mode in mode_list:
    annotate_path = f"/home/mtakahashi/work/lab2021/outputs/0109_transition_pairs/{mode}_annotate"
    f_list = glob(annotate_path+"/*.csv")
    num = 0
    for fpath in f_list:
        num += generate_pos_data(fpath, mode)
    print(f'All {mode} data: {num}')
