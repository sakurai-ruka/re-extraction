# %%
import warnings
warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

import argparse
import os
from re import A
from typing import Optional

import pandas as pd
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning import callbacks, seed_everything
from pytorch_lightning.callbacks import (DeviceStatsMonitor, EarlyStopping,
                                         ModelCheckpoint)
from torch.utils import data
from torch.utils.data import DataLoader
from transformers import BertJapaneseTokenizer

from models import (BertBiLstmModel, BertCnnModel, BertLinearModel,
                    BertLstmModel)


class LabelFileSyntaxError(Exception):
    """ラベルファイルにラベルの不備ある場合に呼ばれるエラー
    """
    pass

def prepare_dummy_dataset() -> pd.DataFrame:
    """ダミー教師データの作成"""
    MAX_TOKEN_LEN = 128
    NUM_DATA = 2
    annotated = ("[ESS]英アストラゼネカのコロナワクチン治験[ESS]、来週再開も＝ＦＴ",
                "[ESS]池袋暴走死傷事故初公判[ESS]　遺族の松永拓也さんが遺影を手に東京地裁へ",
    )
    df = pd.DataFrame(annotated, columns=["annotated"])
    print()
    return df

def error_checker_r(df: pd.DataFrame) -> None:
    """rタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count("<i-t>") == line.count("</i-t>")):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("rラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")

def error_checker_o(df: pd.DataFrame) -> None:
    """oタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count("<i-p>") == line.count("</i-p>")):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("oラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")

def error_checker_e(df: pd.DataFrame) -> None:
    """rタグが正常に付与されているかチェック"""
    flg_error = False
    for i,line in enumerate(df["annotated"].values, start=1):
        if not (line.count("<i-p>") == line.count("</i-p>")):
            flg_error = True
            print(i, line)
    if flg_error:
        raise LabelFileSyntaxError("eラベルファイルにタグの不備があります。出力をヒントに確認してください。例えばタグを閉じ忘れていませんか")

class MyDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        tokenizer,
        max_token_len: int
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 用意したDataFrameの文章(tag除去文)，ラベルのカラム名
        TEXT_COLUMN = "trimmed"
        LABEL_COLUMN = "label"
        data_row = self.data.iloc[idx]
        text = data_row[TEXT_COLUMN]
        labels = data_row[LABEL_COLUMN]
        inputs = self.tokenizer(text, return_tensors="pt",
                    max_length=self.max_token_len, truncation=True, padding="max_length")
        return dict(
                text=text,
                input_ids=inputs["input_ids"].flatten(),
                attention_mask=inputs["attention_mask"].flatten(),
                token_type_ids=inputs["token_type_ids"].flatten(),
            ), torch.tensor(labels)
    
class MyDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_df,
        valid_df,
        test_df,
        max_token_len: int = 256,
        pretrained_model: str = 'cl-tohoku/bert-base-japanese-whole-word-masking',
        train_batchsize: int = 4,
        val_batchsize: int = 2,
        num_workers: int = os.cpu_count(),
    ):
        super().__init__()
        self.train_df = train_df
        self.valid_df = valid_df
        self.test_df = test_df
        self.max_token_len = max_token_len
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
                            pretrained_model,
                            additional_special_tokens=["<i-t>","</i-t>","<i-p>","</i-p>""<i-s>","</i-s>""<i-a>","</i-a>""<i-b>","</i-b>","<e-t>","</e-t>","<e-p>","</e-p>","<e-s>","</e-s>","<e-a>","</e-a>","<e-b>","</e-b>","<o>","</o>"])
        self.num_workers = num_workers
        self.train_batchsize = train_batchsize
        self.val_batchsize = val_batchsize

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.train_ds = MyDataset(self.train_df, self.tokenizer, self.max_token_len)
            self.valid_ds = MyDataset(self.valid_df, self.tokenizer, self.max_token_len)

        if stage == 'test' or stage is None:
            self.test_ds = MyDataset(self.test_df, self.tokenizer, self.max_token_len)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.train_batchsize, shuffle=True, num_workers=self.num_workers)
    def val_dataloader(self):
        # ここでbatch sizeのwarningが出てくる
        return DataLoader(self.valid_ds, batch_size=self.val_batchsize, num_workers=self.num_workers)
    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.val_batchsize, num_workers=self.num_workers)
    def predict_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.val_batchsize, num_workers=self.num_workers)
#%%
def main(args):
    # Load config file
    with open(args.cfg_file) as f:
        cfg = yaml.safe_load(f)
    with open(args.cfg_file) as f:
        print('================ Config file ================')
        print(f.read().strip())
        print('=============================================')

    # Set seed
    seed_everything(cfg['seed'], workers=True)

    df_train = pd.read_pickle("/home/sakurai/git2/lab2021/outputs/train.df.pkl")
    df_test = pd.read_pickle("/home/sakurai/git2/lab2021/outputs/test.df.pkl")

    error_checker_r(df_train)
    error_checker_e(df_train)
    error_checker_o(df_train)
    error_checker_r(df_test)
    error_checker_e(df_test)
    error_checker_o(df_test)

    print(torch.tensor(df_train['label']))
    # データ調整
    train_label = torch.tensor(df_train['label'])
    idx_label_true = torch.sum((train_label == 1)|(train_label == 2)|(train_label == 3)|(train_label == 4)|(train_label == 5)|(train_label == 6)|(train_label == 7)|(train_label == 8)|(train_label == 9)|(train_label == 10)|(train_label == 11), dim=1).bool().tolist()
    df_train = df_train[idx_label_true].reset_index(drop=True)
    print(torch.tensor(df_train['label']))

    test_label = torch.tensor(df_test['label'])
    idx_label_true = torch.sum((test_label == 1)|(test_label == 2)|(test_label == 3)|(test_label == 4)|(test_label == 5)|(test_label == 6)|(test_label == 7)|(test_label == 8)|(test_label == 9)|(test_label == 10)|(test_label == 11), dim=1).bool().tolist()
    df_val = df_test[idx_label_true].reset_index(drop=True)

    logger = pl.loggers.CSVLogger(cfg['output_dir'], name=cfg['exp_name'])

    list_callbacks = []

    # lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval='epoch')
    # list_callbacks.append(lr_monitor)
    # device_stats = DeviceStatsMonitor()
    # list_callbacks.append(device_stats)

    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        min_delta=0.00,
        patience=10,
        verbose=False,
        mode='min'
    )
    list_callbacks.append(early_stop_callback)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        filename="{epoch:02d}-{val_loss:.3f}",
        save_top_k=1,
        mode="min",
    )
    list_callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=list_callbacks,
        # progress_bar_refresh_rate=1,
        **cfg['trainer']
    )

    mydata = MyDataModule(df_train, df_val, df_test, **cfg['dataset_params'])
    if cfg['model_names'] == 'linear':
        try:
            model = BertLinearModel.load_from_checkpoint(checkpoint_path = cfg['pretrained'], tokenizer=mydata.tokenizer, **cfg['model'])
        except KeyError:
            model = BertLinearModel(tokenizer=mydata.tokenizer, **cfg['model'])
    elif cfg['model_names'] == 'lstm':
        try:
            model = BertLstmModel.load_from_checkpoint(checkpoint_path = cfg['pretrained'], tokenizer=mydata.tokenizer, **cfg['model'])
        except KeyError:
            model = BertLstmModel(tokenizer=mydata.tokenizer, **cfg['model'])
    elif cfg['model_names'] == 'bilstm':
        try:
            model = BertBiLstmModel.load_from_checkpoint(checkpoint_path = cfg['pretrained'], tokenizer=mydata.tokenizer, **cfg['model'])
        except KeyError:
            model = BertBiLstmModel(tokenizer=mydata.tokenizer, **cfg['model'])
    # elif cfg['model_names'] == 'cnn': # 実装諦め
    #     model = BertCnnModel(tokenizer=mydata.tokenizer, **cfg['model'])
    
    try:
        cfg['pretrained'] 
        trainer.test(model=model, datamodule=mydata)

    except KeyError:
        trainer.fit(model, mydata)
        trainer.test(model=model, ckpt_path='best', datamodule=mydata)

    # predictions = trainer.predict(model=model, datamodule=mydata)
    with open(args.cfg_file) as f:
        dump = '================ Config file ================\n'+\
                f.read().strip()
    with open(os.path.join(logger.log_dir, "config.txt"), 'w') as outf:
        outf.write(dump)
#%%
# main process
if __name__ == '__main__':
    cfg_file = "/home/sakurai/git2/lab2021/config/token_classification/bilstm_config.yaml"

    parser = argparse.ArgumentParser(description='Evaluate accuracy')
    parser.add_argument('-c', '--cfg-file', type=str, default=cfg_file,
                        help='Config file (.yaml)')
    args = parser.parse_args()
    if args.cfg_file == cfg_file:
        args = parser.parse_args(args=['-c', cfg_file])
    main(args)
# %%
