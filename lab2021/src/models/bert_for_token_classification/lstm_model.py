# %%
import os
from typing import NamedTuple

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .. import visualize_util


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

class BertLstmModel(pl.LightningModule):
    def __init__(
        self,
        optimizer_name: str = 'adam',
        lr: float = 0.01,
        num_labels: int = 2,
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
        self.lstm = nn.LSTM(self.bert.config.hidden_size, self.bert.config.hidden_size, batch_first=True)
        self.token_classifier = nn.Linear(self.bert.config.hidden_size, self.num_labels)

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
            pred_ids = ids[torch.where((preds==1)&(labels!=-100))]
            grand_ids = ids[torch.where((labels==1)&(labels!=-100))]
            pred_tokens = self.tokenizer.convert_ids_to_tokens(pred_ids)
            grand_tokens = self.tokenizer.convert_ids_to_tokens(grand_ids)
            all_tokens = self.tokenizer.convert_ids_to_tokens(ids[labels!=-100])
            self.result += "全トークン\n" \
            f"{self.tokenizer.convert_tokens_to_string(all_tokens)}\n\n" \
            "重要箇所_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens)}\n\n" \
            "重要箇所_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens)}\n\n" \

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
            visualize_util.savefig_roc_curve(t_hist, prob_hist, os.path.join(self.logger.log_dir, 'roc_curve.png'))
            visualize_util.dump_confusion_matrix(t_hist, y_hist, os.path.join(self.logger.log_dir, 'confusion_matrix.csv'))
            visualize_util.dump_classification_report(t_hist, y_hist, os.path.join(self.logger.log_dir, 'classification_report.txt'))
            with open(os.path.join(self.logger.log_dir, 'pred_sentence.txt'), 'w') as f:
                f.write(self.result)
        except AttributeError:
            pass

if __name__ == '__main__':
    model = BertLstmModel()
# %%
