"""
特徴抽出
"""
import argparse
import os.path as osp

import numpy as np
import pandas as pd
import torch
from transformers import BertJapaneseTokenizer, BertModel

def main(args):
    method = args.method

    # ハイパラ
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # 各種モデルの読込
    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')
    model = BertModel.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking')

    model.to(device)
    model.eval()

    root_path = osp.normpath(osp.join(osp.dirname(osp.abspath(__file__)), '..'))
    inf_path = osp.join(root_path, 'outputs/data.df.pickle')
    out_csv_path = osp.join(root_path, f"outputs/extracted_features_{method}.csv")
    out_pkl_path = osp.join(root_path, f"outputs/extracted_features_{method}.df.pickle")

    df = pd.read_pickle(inf_path)


    # tokenize結果出力
    # [ print(tokenizer.decode(x)) for x in tokenized["input_ids"]]

    sentences = df['sentence'].values
    batch_size = 100

    feature_list = []
    for start in range(0, len(sentences), batch_size):

        tokenized = tokenizer(sentences[start: start+batch_size].tolist(), padding=True, truncation=True, return_tensors="pt")
        input_ids = tokenized['input_ids'].to('cuda')

        with torch.no_grad():
            outputs = model(input_ids, return_dict=True)
        pooler_output = outputs["pooler_output"]
        feature_list.append(pooler_output.detach().cpu().numpy())
        break

    import ipdb;ipdb.set_trace()
    feature_array = np.concatenate(feature_list, axis=0)
    num_feature_dims = feature_array.shape[1]
    columns = ['x{}'.format(i) for i in range(num_feature_dims)]
    feature_df = pd.DataFrame(feature_array, columns=columns)
    df_all = pd.concat([df, feature_df], axis=1)
    # df_all.to_csv(out_csv_path, index=None)
    # df_all.to_pickle(out_pkl_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--method', type=str,
        help="feature's method", default='bert')
    args = parser.parse_args()

    main(args)
