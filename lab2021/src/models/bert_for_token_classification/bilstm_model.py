# %%
import os
from typing import NamedTuple
import sys
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
#sys.path.append(os.path.abspath(".."))
from .. import visualize_util
#import visualize_util


class MyBertModel(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str = 'cl-tohoku/bert-base-japanese-whole-word-masking',
    ):
        super().__init__()
        self.model = BertModel.from_pretrained(pretrained_model)
        self.config = self.model.config

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

class BertBiLstmModel(pl.LightningModule):
    def __init__(
        self,
        optimizer_name: str = 'adam',
        lr: float = 0.01,
    #    num_labels: int = 2,
        num_labels: int = 12,

        n_workers: int = os.cpu_count(),
        should_freeze_bert_model: bool = True,
        tokenizer = None,
    ):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.num_labels = num_labels
        self.n_workers = n_workers
        self.should_freeze_bert_model = should_freeze_bert_model
        self.tokenizer = tokenizer

        self.bert = MyBertModel()
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True, bidirectional=True)
        self.token_classifier = nn.Linear(self.bert.config.hidden_size*2, self.num_labels)

        if self.should_freeze_bert_model:
            self.bert.freeze()

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))

    def forward(self, x):
        class LstmDim(NamedTuple):
            """lstmの出力次元数を名前で管理
            """
            b_size: int # x.size[0]:batch
            s_size: int # x.size[1]:sequence
            f_size: int # x.size[2]:feature
        outputs = self.bert(x['input_ids'], x['attention_mask'], x['token_type_ids'])
        out, _ = self.lstm(outputs['last_hidden_state'], None)
        lstm_dim = LstmDim(*out.shape)
        flatten_out = out.reshape(-1, lstm_dim.f_size)
        logits = self.token_classifier(flatten_out)
        logits = logits.reshape(lstm_dim.b_size, lstm_dim.s_size, -1)
        return logits

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)
        loss = F.cross_entropy(y.view(-1, self.num_labels), t.view(-1))
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=t.size(0))
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)
        loss = F.cross_entropy(y.view(-1, self.num_labels), t.view(-1))
        return loss, y

    def validation_step(self, batch, batch_idx):
        loss, _ = self._shared_eval_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, batch_size=batch[1].size(0))
        return loss

    def on_test_epoch_start(self) -> None:
        try:
            self.plot_label_dir = os.path.join(self.logger.log_dir)
            os.makedirs(self.plot_label_dir, exist_ok=True)
        except AttributeError:
            pass
        self.y_hist = []
        self.t_hist = []
        self.prob_hist = []
        self.result = ""

    def test_step(self, batch, batch_idx):
        x, batch_labels = batch
        loss, batch_logits = self._shared_eval_step(batch, batch_idx)

        batch_preds = torch.argmax(batch_logits, dim=2)
        y = batch_preds.view(-1)
        t = batch_labels.view(-1)
        pos_prob = F.softmax(batch_logits, dim=0)[:,:,1].view(-1) # 1の確信度
        ignore_label = -100
        acc = torch.sum(t == y).item() / ( torch.sum(t != ignore_label) * 1.0)

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_epoch=True, on_step=False)

        y = y.cpu().detach().numpy().copy()
        t = t.cpu().detach().numpy().copy()
        pos_prob = pos_prob.cpu().detach().numpy().copy()
        self.y_hist.append(y)
        self.t_hist.append(t)
        self.prob_hist.append(pos_prob)

        for preds, labels, ids in zip(batch_preds, batch_labels, x['input_ids']):
            pred_ids_it = ids[torch.where((preds==1)&(labels!=-100))]
            pred_ids_ip = ids[torch.where((preds==2)&(labels!=-100))]
            pred_ids_is = ids[torch.where((preds==3)&(labels!=-100))]
            pred_ids_ia = ids[torch.where((preds==4)&(labels!=-100))]
            pred_ids_ib = ids[torch.where((preds==5)&(labels!=-100))]
            pred_ids_o = ids[torch.where((preds==6)&(labels!=-100))]
            pred_ids_et = ids[torch.where((preds==7)&(labels!=-100))]
            pred_ids_ep = ids[torch.where((preds==8)&(labels!=-100))]
            pred_ids_es = ids[torch.where((preds==9)&(labels!=-100))]
            pred_ids_ea = ids[torch.where((preds==10)&(labels!=-100))]
            pred_ids_eb = ids[torch.where((preds==11)&(labels!=-100))]

            grand_ids_it = ids[torch.where((labels==1)&(labels!=-100))]
            grand_ids_ip = ids[torch.where((labels==2)&(labels!=-100))]
            grand_ids_is = ids[torch.where((labels==3)&(labels!=-100))]
            grand_ids_ia = ids[torch.where((labels==4)&(labels!=-100))]
            grand_ids_ib = ids[torch.where((labels==5)&(labels!=-100))]
            grand_ids_o = ids[torch.where((labels==6)&(labels!=-100))]
            grand_ids_et = ids[torch.where((labels==7)&(labels!=-100))]
            grand_ids_ep = ids[torch.where((labels==8)&(labels!=-100))]
            grand_ids_es = ids[torch.where((labels==9)&(labels!=-100))]
            grand_ids_ea = ids[torch.where((labels==10)&(labels!=-100))]
            grand_ids_eb = ids[torch.where((labels==11)&(labels!=-100))]

            pred_tokens_it = self.tokenizer.convert_ids_to_tokens(pred_ids_it)
            pred_tokens_ip = self.tokenizer.convert_ids_to_tokens(pred_ids_ip)
            pred_tokens_is = self.tokenizer.convert_ids_to_tokens(pred_ids_is)
            pred_tokens_ia = self.tokenizer.convert_ids_to_tokens(pred_ids_ia)
            pred_tokens_ib = self.tokenizer.convert_ids_to_tokens(pred_ids_ib)
            pred_tokens_o = self.tokenizer.convert_ids_to_tokens(pred_ids_o)
            pred_tokens_et = self.tokenizer.convert_ids_to_tokens(pred_ids_et)
            pred_tokens_ep = self.tokenizer.convert_ids_to_tokens(pred_ids_ep)
            pred_tokens_es = self.tokenizer.convert_ids_to_tokens(pred_ids_es)
            pred_tokens_ea = self.tokenizer.convert_ids_to_tokens(pred_ids_ea)
            pred_tokens_eb = self.tokenizer.convert_ids_to_tokens(pred_ids_eb)

            grand_tokens_it = self.tokenizer.convert_ids_to_tokens(grand_ids_it)
            grand_tokens_ip = self.tokenizer.convert_ids_to_tokens(grand_ids_ip)
            grand_tokens_is = self.tokenizer.convert_ids_to_tokens(grand_ids_is)
            grand_tokens_ia = self.tokenizer.convert_ids_to_tokens(grand_ids_ia)
            grand_tokens_ib = self.tokenizer.convert_ids_to_tokens(grand_ids_ib)
            grand_tokens_o = self.tokenizer.convert_ids_to_tokens(grand_ids_o)
            grand_tokens_et = self.tokenizer.convert_ids_to_tokens(grand_ids_et)
            grand_tokens_ep = self.tokenizer.convert_ids_to_tokens(grand_ids_ep)
            grand_tokens_es = self.tokenizer.convert_ids_to_tokens(grand_ids_es)
            grand_tokens_ea = self.tokenizer.convert_ids_to_tokens(grand_ids_ea)
            grand_tokens_eb = self.tokenizer.convert_ids_to_tokens(grand_ids_eb)

            all_tokens = self.tokenizer.convert_ids_to_tokens(ids[labels!=-100])
            print(all_tokens)
            self.result += "全トークン\n" \
            f"{self.tokenizer.convert_tokens_to_string(all_tokens)}\n\n" \
            "重要箇所(正解：対象)_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens_o)}\n\n" \
            "重要箇所(予測：対象)_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens_o)}\n\n" \
            "重要箇所(正解：商品特有情報)_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens_it)}\n\n" \
            "重要箇所(予測：商品特有情報)_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens_it)}\n\n" \
            "重要箇所(正解：価格情報)_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens_ip)}\n\n" \
            "重要箇所(予測：価格情報)_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens_ip)}\n\n" \
            "重要箇所(正解：量・状態情報)_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens_is)}\n\n" \
            "重要箇所(予測：量・状態情報)_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens_is)}\n\n" \
            "重要箇所(正解：品質情報)_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens_ia)}\n\n" \
            "重要箇所(予測：品質情報)_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens_ia)}\n\n" \
            "重要箇所(正解：他情報)_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens_ib)}\n\n" \
            "重要箇所(予測：他情報)_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens_ib)}\n\n" \
            "重要箇所(正解：商品特有感想)_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens_et)}\n\n" \
            "重要箇所(予測：商品特有感想)_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens_et)}\n\n" \
            "重要箇所(正解：価格感想)_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens_ep)}\n\n" \
            "重要箇所(予測：価格感想)_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens_ep)}\n\n" \
            "重要箇所(正解：量・状態感想)_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens_es)}\n\n" \
            "重要箇所(予測：量・状態感想)_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens_es)}\n\n" \
            "重要箇所(正解：品質感想)_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens_ea)}\n\n" \
            "重要箇所(予測：品質感想)_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens_ea)}\n\n" \
            "重要箇所(正解：その他感想)_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens_eb)}\n\n" \
            "重要箇所(予測：その他感想)_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens_eb)}\n\n" \


        return {'loss': loss, 'y': y, 't': t}

    def on_test_epoch_end(self)-> None:
        prob_hist = np.concatenate(self.prob_hist)
        y_hist = np.concatenate(self.y_hist)
        t_hist = np.concatenate(self.t_hist)

        prob_hist = prob_hist[np.where(t_hist!=-100)]
        y_hist = y_hist[np.where(t_hist!=-100)]
        t_hist = t_hist[np.where(t_hist!=-100)]

        try:
            visualize_util.dump_preds(t_hist, y_hist, os.path.join(self.logger.log_dir, 'preds.csv'))
            #visualize_util.savefig_roc_curve(t_hist, prob_hist, os.path.join(self.logger.log_dir, 'roc_curve.png'))
            visualize_util.dump_confusion_matrix(t_hist, y_hist, os.path.join(self.logger.log_dir, 'confusion_matrix.csv'))
            visualize_util.dump_classification_report(t_hist, y_hist, os.path.join(self.logger.log_dir, 'classification_report.txt'))
            with open(os.path.join(self.logger.log_dir, 'pred_sentence.txt'), 'w') as f:
                f.write(self.result)
        except AttributeError:
            pass

if __name__ == '__main__':
    model = BertBiLstmModel()
# %%
