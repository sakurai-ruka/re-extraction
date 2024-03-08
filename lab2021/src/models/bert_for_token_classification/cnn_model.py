# %%
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import os
from typing import Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from transformers import (BertForTokenClassification, BertJapaneseTokenizer,
                          BertModel, BertTokenizer, logging, BertConfig)
from transformers.utils.dummy_pt_objects import PreTrainedModel

import torch.nn as nn
import sklearn_crfsuite

class MyBertModel(pl.LightningModule):
    def __init__(
        self,
        pretrained_model: str = 'cl-tohoku/bert-base-japanese-whole-word-masking',
    ):
        super().__init__()
        self.tokenizer = BertJapaneseTokenizer.from_pretrained(
                        pretrained_model,
                        additional_special_tokens=["[ESS]"])
        self.model = BertModel.from_pretrained(pretrained_model)
        self.config = self.model.config

    def forward(self, input_ids, attention_mask, token_type_ids):
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids)

class BertCnnModel(pl.LightningModule):
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

    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            self.bert = MyBertModel()
            self.cnn = nn.Conv1d(self.bert.config.hidden_size, self.bert.config.hidden_size, kernel_size=2, stride=1)
            self.pool = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
            self.crf = sklearn_crfsuite.CRF(
                            algorithm='lbfgs',
                            c1=0.1,
                            c2=0.1,
                            max_iterations=100,
                            all_possible_transitions=True
            )
            # self.linear = nn.Linear(self.bert.config.hidden_size*2, self.num_labels)

            if self.should_freeze_bert_model:
                self.bert.freeze()

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))

    def forward(self, x):
        outputs = self.bert(x['input_ids'], x['attention_mask'], x['token_type_ids'])
        last_hidden_state = outputs['last_hidden_state'].permute(0, 2, 1)
        cnn_embeddings = self.pool(self.cnn(last_hidden_state))
        # logits = self.linear(sequence_output)
        X = [pd.DataFrame(embedding.cpu()).to_dict() for embedding in cnn_embeddings]
        return logits

    def training_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)
        loss = F.cross_entropy(y, t.view(-1))
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True, batch_size=t.size(0))
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x, t = batch
        y = self.forward(x)
        loss = F.cross_entropy(y, t.view(-1))
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
        self.result = ""

    def test_step(self, batch, batch_idx):
        x, batch_labels = batch
        loss, batch_logits = self._shared_eval_step(batch, batch_idx)

        batch_preds = torch.argmax(batch_logits, dim=1)
        y = batch_preds.view(-1)
        t = batch_labels.view(-1)
        ignore_label = -100
        acc = torch.sum(t == y).item() / ( torch.sum(t != ignore_label) * 1.0)

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_epoch=True, on_step=False)

        y = y.cpu().detach().numpy().copy()
        t = t.cpu().detach().numpy().copy()
        self.y_hist.append(y)
        self.t_hist.append(t)

        for preds, labels, ids, mask in zip([batch_preds[:128], batch_preds[128:]], batch_labels, x['input_ids'], x['attention_mask']):
            mask = mask.bool()
            pred_ids = ids[torch.where(preds[mask]==1)]
            grand_ids = ids[torch.where(labels[mask]==1)]
            pred_tokens = self.tokenizer.convert_ids_to_tokens(pred_ids)
            grand_tokens = self.tokenizer.convert_ids_to_tokens(grand_ids)
            all_tokens = self.tokenizer.convert_ids_to_tokens(ids[mask])
            self.result += "全トークン\n" \
            f"{self.tokenizer.convert_tokens_to_string(all_tokens)}\n\n" \
            "重要箇所_grand\n" \
            f"{self.tokenizer.convert_tokens_to_string(grand_tokens)}\n\n" \
            "重要箇所_pred\n" \
            f"{self.tokenizer.convert_tokens_to_string(pred_tokens)}\n\n" \

        return {'loss': loss, 'y': y, 't': t}

    def on_test_epoch_end(self)-> None:
        y_hist = np.concatenate(self.y_hist)
        t_hist = np.concatenate(self.t_hist)

        target_names = ["Not important", "important"]
        report = classification_report(
            t_hist, y_hist, target_names=target_names, labels=range(self.num_labels),
            output_dict=False, zero_division=0)
            # output_dict=True, zero_division=0)

        C = confusion_matrix(t_hist, y_hist, labels=range(self.num_labels))
        df = pd.DataFrame(C, columns=target_names, index=target_names)

        # if 'accuracy' in report:
        #     self.log('test_accuracy', report['accuracy'], prog_bar=False, on_epoch=True)
        # else:
        #     self.log('test_accuracy', report['micro avg']['precision'], prog_bar=False, on_epoch=True)

        try:
            with open(os.path.join(self.logger.log_dir, 'pred_sentence.txt'), 'w') as f:
                f.write(self.result)
            with open(os.path.join(self.logger.log_dir, 'classification_report.txt'), 'w') as f:
                f.write(report)
            np.savetxt(os.path.join(self.logger.log_dir, 'preds.txt'), y_hist, fmt='%d')
            df.to_csv(os.path.join(self.logger.log_dir, 'confusion_matrix.csv'))
        except AttributeError:
            pass

if __name__ == '__main__':
    model = BertCnnModel()
    model.setup()
# %%
