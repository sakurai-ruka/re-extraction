# %%
import os

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from transformers import BertForTokenClassification

from .. import visualize_util


class BertLinearModel(pl.LightningModule):
    def __init__(
        self,
        optimizer_name: str = 'adam',
        lr: float = 0.01,
        num_labels: int = 2,
        n_workers: int = 20,
        pretrained_model: str = 'cl-tohoku/bert-base-japanese-whole-word-masking',
        should_freeze_bert_model: bool = True,
        tokenizer = None,
    ):
        super().__init__()
        self.optimizer_name = optimizer_name
        self.lr = lr
        self.num_labels = num_labels
        self.n_workers = n_workers
        self.tokenizer = tokenizer

        self.model = BertForTokenClassification.from_pretrained(
            pretrained_model)
        
        if not should_freeze_bert_model:
            return

        for param in self.model.bert.parameters():
            param.requires_grad = False
        self.model.bert.eval()

    def forward(self, x):
        return self.model(input_ids=x['input_ids'], attention_mask=x['attention_mask'])

    def training_step(self, batch, batch_idx):
        x, t = batch
        outputs = self.forward(x)
        y = outputs.logits
        loss = F.cross_entropy(y.view(-1, self.num_labels), t.view(-1))
        self.log('train_loss', loss, prog_bar=True, on_epoch=True, on_step=True)
        return loss

    def _shared_eval_step(self, batch, batch_idx):
        x, t = batch
        outputs = self.forward(x)
        logits = outputs.logits
        y = logits.view(-1, self.num_labels)
        loss = F.cross_entropy(y, t.view(-1))
        return loss, logits

    def validation_step(self, batch, batch_idx):
        loss, _ = self._shared_eval_step(batch, batch_idx)
        self.log('val_loss', loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def on_test_epoch_start(self) -> None:
        try:
            self.plot_label_dir = os.path.join(self.logger.log_dir, 'predicted')
            os.makedirs(self.plot_label_dir, exist_ok=True)
        except AttributeError:
            pass
        self.y_hist = []
        self.t_hist = []
        self.logits_hist = []
        self.result = ""

    def test_step(self, batch, batch_idx):
        x, batch_labels = batch
        loss, batch_logits = self._shared_eval_step(batch, batch_idx)

        batch_preds = torch.argmax(batch_logits, dim=2)
        y = batch_preds.view(-1)
        t = batch_labels.view(-1)
        batch_logits = batch_logits.view(-1, self.num_labels)
        true_pred = F.softmax(batch_logits)[:, 1] # 1の確信度
        ignore_label = -100
        acc = torch.sum(t == y).item() / ( torch.sum(t != ignore_label) * 1.0)

        metrics = {"test_acc": acc, "test_loss": loss}
        self.log_dict(metrics, on_epoch=True, on_step=False)

        y = y.cpu().detach().numpy().copy()
        t = t.cpu().detach().numpy().copy()
        batch_logits_ = true_pred.cpu().detach().numpy().copy()

        self.y_hist.append(y)
        self.t_hist.append(t)
        self.logits_hist.append(batch_logits_)

        for preds, labels, ids, mask in zip(batch_preds, batch_labels, x['input_ids'], x['attention_mask']):
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
        pred_hist = np.concatenate(self.logits_hist)
        y_hist = np.concatenate(self.y_hist)
        t_hist = np.concatenate(self.t_hist)

        pred_hist = pred_hist[np.where(t_hist!=-100)]
        y_hist = y_hist[np.where(t_hist!=-100)]
        t_hist = t_hist[np.where(t_hist!=-100)]

        try:
            visualize_util.dump_preds(t_hist, y_hist, os.path.join(self.logger.log_dir, 'preds.csv'))
            visualize_util.savefig_roc_curve(t_hist, pred_hist, os.path.join(self.logger.log_dir, 'roc_curve.png'))
            visualize_util.dump_confusion_matrix(t_hist, y_hist, os.path.join(self.logger.log_dir, 'confusion_matrix.csv'))
            visualize_util.dump_classification_report(t_hist, y_hist, os.path.join(self.logger.log_dir, 'classification_report.txt'))
            with open(os.path.join(self.logger.log_dir, 'pred_sentence.txt'), 'w') as f:
                f.write(self.result)
        except AttributeError:
            pass

    def configure_optimizers(self):
        if self.optimizer_name == 'adam':
            return torch.optim.Adam(self.parameters(), lr=(self.lr or self.learning_rate))
# %%
