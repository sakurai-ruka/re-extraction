"""
タグ付けされた文章からtoken classificationのラベルを作成
次はtoken_classification, sequence_classification

ex.
[ESS]吾輩[ESS] は 猫 で ある
label: token毎のラベル  like. 1 0 0 0 0
sequense_label: 文中にESS区間があるか無いか  like. 1
"""
# %%
MODE = "test"
#TOKEN = "[ESS]"
TOKEN_it = "<i-t>"
TOKEN_iet = "</i-t>"
TOKEN_ip = "<i-p>"
TOKEN_iep = "</i-p>"
TOKEN_is = "<i-s>"
TOKEN_ies = "</i-s>"
TOKEN_ia = "<i-a>"
TOKEN_iea = "</i-a>"
TOKEN_ib = "<i-b>"
TOKEN_ieb = "</i-b>"
TOKEN_et = "<e-t>"
TOKEN_eet = "</e-t>"
TOKEN_ep = "<e-p>"
TOKEN_eep = "</e-p>"
TOKEN_es = "<e-s>"
TOKEN_ees = "</e-s>"
TOKEN_ea = "<e-a>"
TOKEN_eea = "</e-a>"
TOKEN_eb = "<e-b>"
TOKEN_eeb = "</e-b>"
TOKEN_o = "<o>"
TOKEN_oe = "</o>"

import argparse
#from operator import or_
import os
from glob import glob
from re import A
from tkinter import E

import numpy as np
import pandas as pd
import torch
import yaml
from transformers import BertJapaneseTokenizer, logging

logging.set_verbosity_info()

class LabelFileSyntaxError(Exception):
    """ラベルの不備がある場合に呼ばれる例外"""
    pass

def prepare_dummy_dataset() -> pd.DataFrame:
    """ダミー教師データの作成"""
    annotated = ("[ESS]英アストラゼネカのコロナワクチン治験[ESS]、来週再開も＝ＦＴ",
                "[ESS]池袋暴走死傷事故初公判[ESS]　遺族の松永拓也さんが遺影を手に東京地裁へ",
                "【米大統領選】バイデン氏陣営、政権移行に本腰　トランプ氏の「不正」主張に行き詰まり感　「集計機の不正」主張も不発か",
    )
    return pd.DataFrame(annotated, columns=["annotated"])

def error_checker_it(df: pd.DataFrame) -> None:
    """itタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count(TOKEN_it) == line.count(TOKEN_iet)):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("itラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")

def error_checker_ip(df: pd.DataFrame) -> None:
    """ipタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count(TOKEN_ip) == line.count(TOKEN_iep)):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("ipラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")

def error_checker_is(df: pd.DataFrame) -> None:
    """isタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count(TOKEN_is) == line.count(TOKEN_ies)):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("isラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")

def error_checker_ia(df: pd.DataFrame) -> None:
    """iaタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count(TOKEN_ia) == line.count(TOKEN_iea)):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("iaラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")

def error_checker_ib(df: pd.DataFrame) -> None:
    """ibタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count(TOKEN_ib) == line.count(TOKEN_ieb)):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("ibラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")

def error_checker_o(df: pd.DataFrame) -> None:
    """oタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count(TOKEN_o) == line.count(TOKEN_oe)):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("oラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")

def error_checker_et(df: pd.DataFrame) -> None:
    """etタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count(TOKEN_et) == line.count(TOKEN_eet)):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("etラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")

def error_checker_ep(df: pd.DataFrame) -> None:
    """epタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count(TOKEN_ep) == line.count(TOKEN_eep)):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("epラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")
def error_checker_es(df: pd.DataFrame) -> None:
    """esタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count(TOKEN_es) == line.count(TOKEN_ees)):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("esラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")
def error_checker_ea(df: pd.DataFrame) -> None:
    """etタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count(TOKEN_ea) == line.count(TOKEN_eea)):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("eaラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")
def error_checker_eb(df: pd.DataFrame) -> None:
    """ebタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count(TOKEN_eb) == line.count(TOKEN_eeb)):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("ebラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")

# %%
if __name__ == '__main__':
    TOKENIZE_BATCHSIZE = 1

    # ラベルファイルをまとめる
    if MODE == 'train':
        DATA_ROOT = "/home/sakurai/git2/lab2021/data/token_classification"
        fnames = glob(DATA_ROOT + "/reniku.txt")
    elif MODE =='test':
        DATA_ROOT = "/home/sakurai/git2/lab2021/data/token_classification"
        fnames = glob(DATA_ROOT + "/retest.txt")
    # fnames.append('/home/mtakahashi/work/lab2021/data/token_classification/英&アストラゼネカ.csv')
    list_df = [pd.read_csv(fname, names=["annotated"]) for fname in fnames]
    print(list_df)
    df = pd.concat(list_df).reset_index(drop=True)
    # df = prepare_dummy_dataset()
    error_checker_o(df)
    error_checker_it(df)
    error_checker_ip(df)
    error_checker_is(df)
    error_checker_ia(df)
    error_checker_ib(df)
    error_checker_et(df)
    error_checker_ep(df)
    error_checker_es(df)
    error_checker_ea(df)
    error_checker_eb(df)


    tokenizer = BertJapaneseTokenizer.from_pretrained('cl-tohoku/bert-base-japanese-whole-word-masking',
                    additional_special_tokens=[TOKEN_o,TOKEN_oe,TOKEN_it,TOKEN_iet,TOKEN_ip,TOKEN_iep,TOKEN_is,TOKEN_ies,TOKEN_ia,TOKEN_iea,TOKEN_ib,TOKEN_ieb,TOKEN_et,TOKEN_eet,TOKEN_ep,TOKEN_eep,TOKEN_es,TOKEN_ees,TOKEN_ea,TOKEN_eea,TOKEN_eb,TOKEN_eeb])
    token_id_essential_tag_o = tokenizer.convert_tokens_to_ids(TOKEN_o)
    token_id_essential_tag_oe = tokenizer.convert_tokens_to_ids(TOKEN_oe)
    token_id_essential_tag_it = tokenizer.convert_tokens_to_ids(TOKEN_it)
    token_id_essential_tag_iet = tokenizer.convert_tokens_to_ids(TOKEN_iet)
    token_id_essential_tag_ip = tokenizer.convert_tokens_to_ids(TOKEN_ip)
    token_id_essential_tag_iep = tokenizer.convert_tokens_to_ids(TOKEN_iep)
    token_id_essential_tag_is = tokenizer.convert_tokens_to_ids(TOKEN_is)
    token_id_essential_tag_ies = tokenizer.convert_tokens_to_ids(TOKEN_ies)
    token_id_essential_tag_ia = tokenizer.convert_tokens_to_ids(TOKEN_ia)
    token_id_essential_tag_iea = tokenizer.convert_tokens_to_ids(TOKEN_iea)
    token_id_essential_tag_ib = tokenizer.convert_tokens_to_ids(TOKEN_ib)
    token_id_essential_tag_ieb = tokenizer.convert_tokens_to_ids(TOKEN_ieb)
    token_id_essential_tag_et = tokenizer.convert_tokens_to_ids(TOKEN_et)
    token_id_essential_tag_eet = tokenizer.convert_tokens_to_ids(TOKEN_eet)
    token_id_essential_tag_ep = tokenizer.convert_tokens_to_ids(TOKEN_ep)
    token_id_essential_tag_eep = tokenizer.convert_tokens_to_ids(TOKEN_eep)
    token_id_essential_tag_es = tokenizer.convert_tokens_to_ids(TOKEN_es)
    token_id_essential_tag_ees = tokenizer.convert_tokens_to_ids(TOKEN_ees)
    token_id_essential_tag_ea = tokenizer.convert_tokens_to_ids(TOKEN_ea)
    token_id_essential_tag_eea = tokenizer.convert_tokens_to_ids(TOKEN_eea)
    token_id_essential_tag_eb = tokenizer.convert_tokens_to_ids(TOKEN_eb)
    token_id_essential_tag_eeb = tokenizer.convert_tokens_to_ids(TOKEN_eeb)

    labels = []
    
    for batch_number, batch_df in df["annotated"].groupby(np.arange(len(df)) // TOKENIZE_BATCHSIZE):
        inputs = tokenizer(batch_df.values.tolist(), return_tensors="pt",
                            max_length=512//2, truncation=True, padding="max_length")
        #print(batch_df)
        #print(tokenizer.tokenize("届いたときは<i-a>小さい</i-a>かな？と思いましたが、クリスマスからお正月まで食べられました！<o>お肉</o>が<i-t>柔らかく</i-t>、<i-t>噛みしめるほど味が出てきて</i-t>とても<e-t>美味しかった</e-t>です。解凍の仕方や食べ方の説明も入っていて、添付のソースも<e-t>美味しかった</e-t>です。試しにと思って買って、<e-b>大満足</e-b>でした！ また購入したいです！"))
 
        #print(type(batch_df))
        #batch_df2 = batch_df.astype(str)
        #print(type(batch_df2))
        #print(batch_df.count(TOKEN_r))
        attention_mask = inputs["attention_mask"]
        
        idx_o, ess_idx_o = torch.where(inputs["input_ids"]==token_id_essential_tag_o)
        idx_oe, ess_idx_oe = torch.where(inputs["input_ids"]==token_id_essential_tag_oe)
        idx_it, ess_idx_it = torch.where(inputs["input_ids"]==token_id_essential_tag_it)
        idx_iet, ess_idx_iet = torch.where(inputs["input_ids"]==token_id_essential_tag_iet)
        idx_ip, ess_idx_ip = torch.where(inputs["input_ids"]==token_id_essential_tag_ip)
        idx_iep, ess_idx_iep = torch.where(inputs["input_ids"]==token_id_essential_tag_iep)
        idx_is, ess_idx_is = torch.where(inputs["input_ids"]==token_id_essential_tag_is)
        idx_ies, ess_idx_ies = torch.where(inputs["input_ids"]==token_id_essential_tag_ies)
        idx_ia, ess_idx_ia = torch.where(inputs["input_ids"]==token_id_essential_tag_ia)
        idx_iea, ess_idx_iea = torch.where(inputs["input_ids"]==token_id_essential_tag_iea)
        idx_ib, ess_idx_ib = torch.where(inputs["input_ids"]==token_id_essential_tag_ib)
        idx_ieb, ess_idx_ieb = torch.where(inputs["input_ids"]==token_id_essential_tag_ieb)
        idx_et, ess_idx_et = torch.where(inputs["input_ids"]==token_id_essential_tag_et)
        idx_eet, ess_idx_eet = torch.where(inputs["input_ids"]==token_id_essential_tag_eet)
        idx_ep, ess_idx_ep = torch.where(inputs["input_ids"]==token_id_essential_tag_ep)
        idx_eep, ess_idx_eep = torch.where(inputs["input_ids"]==token_id_essential_tag_eep)
        idx_es, ess_idx_es = torch.where(inputs["input_ids"]==token_id_essential_tag_es)
        idx_ees, ess_idx_ees = torch.where(inputs["input_ids"]==token_id_essential_tag_ees)
        idx_ea, ess_idx_ea = torch.where(inputs["input_ids"]==token_id_essential_tag_ea)
        idx_eea, ess_idx_eea = torch.where(inputs["input_ids"]==token_id_essential_tag_eea)
        idx_eb, ess_idx_eb = torch.where(inputs["input_ids"]==token_id_essential_tag_eb)
        idx_eeb, ess_idx_eeb = torch.where(inputs["input_ids"]==token_id_essential_tag_eeb)

        batch_labels = torch.tensor([[0] * inputs["input_ids"].size(1) for _ in range(inputs["input_ids"].size(0))])
        print(batch_labels)
        # rタグが無い状況のtoken label作成
        # 開始タグを<r-t>、終了タグを</r-t>とする
        if len(idx_it):
            it_it = iter(ess_idx_it.numpy()) # タグの位置インデックスを開始・終了の2個ずつ取りたいので
            it_iet = iter(ess_idx_iet.numpy()) 
            for jr, start, end in zip(idx_it, it_it, it_iet):
                batch_labels[jr, start:end-1] = 1
        # 開始タグを<r-p>、終了タグを</r-p>とする
        if len(idx_ip):
            it_ip = iter(ess_idx_ip.numpy()) # タグの位置インデックスを開始・終了の2個ずつ取りたいので
            it_iep = iter(ess_idx_iep.numpy()) 
            for jr, start, end in zip(idx_ip, it_ip, it_iep):
                batch_labels[jr, start:end-1] = 2
        # 開始タグを<r-s>、終了タグを</r-s>とする
        if len(idx_is):
            it_is = iter(ess_idx_is.numpy()) # タグの位置インデックスを開始・終了の2個ずつ取りたいので
            it_ies = iter(ess_idx_ies.numpy()) 
            for jr, start, end in zip(idx_is, it_is, it_ies):
                batch_labels[jr, start:end-1] = 3
        # 開始タグを<r-a>、終了タグを</r-a>とする
        if len(idx_ia):
            it_ia = iter(ess_idx_ia.numpy()) # タグの位置インデックスを開始・終了の2個ずつ取りたいので
            it_iea = iter(ess_idx_iea.numpy()) 
            for jr, start, end in zip(idx_ia, it_ia, it_iea):
                batch_labels[jr, start:end-1] = 4
        # 開始タグを<r-b>、終了タグを</r-b>とする
        if len(idx_ib):
            it_ib = iter(ess_idx_ib.numpy()) # タグの位置インデックスを開始・終了の2個ずつ取りたいので
            it_ieb = iter(ess_idx_ieb.numpy()) 
            for jr, start, end in zip(idx_ib, it_ib, it_ieb):
                batch_labels[jr, start:end-1] = 5
       
        # oタグが無い状況のtoken label作成
        if len(idx_o):
            it_o = iter(ess_idx_o.numpy()) # ESSタグの位置インデックスを開始・終了の2個ずつ取りたいので
            it_oe = iter(ess_idx_oe.numpy()) 
            for jo, start, end in zip(idx_o, it_o, it_oe):
                batch_labels[jo, start:end-1] = 6
        # eタグが無い状況のtoken label作成
        if len(idx_et):
            it_et = iter(ess_idx_et.numpy()) # ESSタグの位置インデックスを開始・終了の2個ずつ取りたいので
            it_eet = iter(ess_idx_eet.numpy()) 
            for je, start, end in zip(idx_et, it_et, it_eet):
                batch_labels[je, start:end-1] = 7
        # eタグが無い状況のtoken label作成
        if len(idx_ep):
            it_ep = iter(ess_idx_ep.numpy()) # ESSタグの位置インデックスを開始・終了の2個ずつ取りたいので
            it_eep = iter(ess_idx_eep.numpy()) 
            for je, start, end in zip(idx_ep, it_ep, it_eep):
                batch_labels[je, start:end-1] = 8
        # eタグが無い状況のtoken label作成
        if len(idx_es):
            it_es = iter(ess_idx_es.numpy()) # ESSタグの位置インデックスを開始・終了の2個ずつ取りたいので
            it_ees = iter(ess_idx_ees.numpy()) 
            for je, start, end in zip(idx_es, it_es, it_ees):
                batch_labels[je, start:end-1] = 9
        # eタグが無い状況のtoken label作成
        if len(idx_ea):
            it_ea = iter(ess_idx_ea.numpy()) # ESSタグの位置インデックスを開始・終了の2個ずつ取りたいので
            it_eea = iter(ess_idx_eea.numpy()) 
            for je, start, end in zip(idx_ea, it_ea, it_eea):
                batch_labels[je, start:end-1] = 10
        # eタグが無い状況のtoken label作成
        if len(idx_eb):
            it_eb = iter(ess_idx_eb.numpy()) # ESSタグの位置インデックスを開始・終了の2個ずつ取りたいので
            it_eeb = iter(ess_idx_eeb.numpy()) 
            for je, start, end in zip(idx_eb, it_eb, it_eeb):
                batch_labels[je, start:end-1] = 11

        print(batch_labels)
        batch_labels_dl = batch_labels
        #タグの文をずらす
        shift = 1
        e_labels = torch.cat((ess_idx_iet,ess_idx_iep,ess_idx_ies,ess_idx_iea,ess_idx_ieb,ess_idx_oe,ess_idx_eet,ess_idx_eep,ess_idx_ees,ess_idx_eea,ess_idx_eeb),-1)
        for i in range(128):
            la = batch_labels[:,i]
            if la == 1:
                batch_labels[:,i-shift] = 1
                batch_labels[:,i] = 0
            if la == 2:
                batch_labels[:,i-shift] = 2
                batch_labels[:,i] = 0
            if la == 3:
                batch_labels[:,i-shift] = 3
                batch_labels[:,i] = 0
            if la == 4:
                batch_labels[:,i-shift] = 4
                batch_labels[:,i] = 0
            if la == 5:
                batch_labels[:,i-shift] = 5
                batch_labels[:,i] = 0
            if la == 6:
                batch_labels[:,i-shift] = 6
                batch_labels[:,i] = 0
            if la == 7:
                batch_labels[:,i-shift] = 7
                batch_labels[:,i] = 0
            if la == 8:
                batch_labels[:,i-shift] = 8
                batch_labels[:,i] = 0
            if la == 9:
                batch_labels[:,i-shift] = 9
                batch_labels[:,i] = 0
            if la == 10:
                batch_labels[:,i-shift] = 10
                batch_labels[:,i] = 0
            if la == 11:
                batch_labels[:,i-shift] = 11
                batch_labels[:,i] = 0
            if i in e_labels:
                shift += 2
        batch_labels = torch.roll(batch_labels, 1)
        # SEPタグ1つ分 attention_maskを左シフト
        attention_mask = torch.roll(attention_mask, -1)
        attention_mask[:, -1:] = 0
        # タグを消す分 attention_maskを左シフト
        if len(idx_it):
            for i in range(len(idx_it)):
                attention_mask = torch.roll(attention_mask, -2)
                attention_mask[:, -2:] = 0
        if len(idx_ip):
            for i in range(len(idx_ip)):
                attention_mask = torch.roll(attention_mask, -2)
                attention_mask[:, -2:] = 0
        if len(idx_is):
            for i in range(len(idx_is)):
                attention_mask = torch.roll(attention_mask, -2)
                attention_mask[:, -2:] = 0
        if len(idx_ia):
            for i in range(len(idx_ia)):
                attention_mask = torch.roll(attention_mask, -2)
                attention_mask[:, -2:] = 0
        if len(idx_ib):
            for i in range(len(idx_ib)):
                attention_mask = torch.roll(attention_mask, -2)
                attention_mask[:, -2:] = 0

        if len(idx_o):
            for i in range(len(idx_o)):
                attention_mask = torch.roll(attention_mask, -2)
                attention_mask[:, -2:] = 0
        if len(idx_et):
            for i in range(len(idx_et)):
                attention_mask = torch.roll(attention_mask, -2)
                attention_mask[:, -2:] = 0
        if len(idx_ep):
            for i in range(len(idx_ep)):
                attention_mask = torch.roll(attention_mask, -2)
                attention_mask[:, -2:] = 0
        if len(idx_es):
            for i in range(len(idx_es)):
                attention_mask = torch.roll(attention_mask, -2)
                attention_mask[:, -2:] = 0
        if len(idx_ea):
            for i in range(len(idx_ea)):
                attention_mask = torch.roll(attention_mask, -2)
                attention_mask[:, -2:] = 0
        if len(idx_eb):
            for i in range(len(idx_eb)):
                attention_mask = torch.roll(attention_mask, -2)
                attention_mask[:, -2:] = 0

        batch_labels[~attention_mask.bool()] = -100 # pad+SEP部分
        #print(batch_labels)
        # SEPタグ1つ分 attention_maskを右シフト
        attention_mask = torch.roll(attention_mask, 1)
        attention_mask[:, :1] = 1
        #print(batch_labels)
        batch_labels[:, 0] = -100 # CLSタグ
        #print(batch_labels)
        labels.append(batch_labels)


    df["trimmed"] = df["annotated"].str.replace(r"\<o>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\</o>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\<i-t>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\</i-t>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\<i-p>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\</i-p>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\<i-s>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\</i-s>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\<i-a>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\</i-a>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\<i-b>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\</i-b>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\<e-t>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\</e-t>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\<e-p>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\</e-p>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\<e-s>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\</e-s>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\<e-a>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\</e-a>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\<e-b>","", regex=True)
    df["trimmed"] = df["trimmed"].str.replace(r"\</e-b>","", regex=True)


    labels = torch.cat(labels)
    df["label"] = pd.DataFrame({'label': labels.tolist()})
    df["sequence_label"] = pd.DataFrame({'sequence_label': (labels.max(dim=1).values>0).int().tolist()})
    df.to_pickle(os.path.join('/home/sakurai/git2/lab2021/outputs', f"{MODE}.df.pkl"))
# %%
