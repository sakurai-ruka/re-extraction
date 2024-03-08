"""
不正データの除去
1回回せば終わり
"""
import os.path as osp

import pandas as pd

root_path = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
inf_path = osp.join(root_path, 'data/test copy.csv')
out_csv_path = osp.join(root_path, 'outputs/data.csv')
out_pkl_path = osp.join(root_path, 'outputs/data.df.pickle')

result = []
with open(inf_path, 'r') as f:
    for line in f:
        contents = line.strip().split(',')
        sentence = '、'.join(contents[4:])
        if sentence == '':
            continue

        contents = contents[:4]
        contents.append(sentence)
        result.append(contents)

df = pd.DataFrame(result, columns=['id', 'category', 'date', 'sentence_num', 'sentence'])

# df.to_csv(out_csv_path, index=None)
# df.to_pickle(out_pkl_path)

pd.read_pickle(out_pkl_path)