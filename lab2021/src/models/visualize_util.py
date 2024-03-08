import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (auc, classification_report, confusion_matrix,
                             roc_curve)


def savefig_roc_curve(t_hist, y_hist, fpath):
    fpr, tpr, thresholds = roc_curve(t_hist, y_hist)
    roc_auc = auc(fpr, tpr)

    fig = plt.figure()
    ax = fig.add_subplot(111,
            xlim=(0,1),
            ylim=(0,1),
            xlabel='FPR: False positive rate',
            ylabel='TPR: True positive rate',
        )
    ax.plot(fpr, tpr, label=f'ROC curve (area = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.legend(loc='lower right')
    ax.grid()
    fig.savefig(fpath)
    plt.close()

def dump_confusion_matrix(t_hist, y_hist, fpath):
    target_names = ["Not important", "important"]
    C = confusion_matrix(t_hist, y_hist, labels=[0,1])
    df = pd.DataFrame(C, columns=target_names, index=target_names)
    df.to_csv(fpath)

def dump_classification_report(t_hist, y_hist, fpath):
    target_names = ["Not important", "important"]
    report = classification_report(
        t_hist, y_hist, target_names=target_names, labels=[0,1],
        output_dict=False, zero_division=0)

    with open(fpath, 'w') as f:
        f.write(report)

def dump_preds(t_hist, y_hist, fpath):
    t_hist = t_hist.reshape(-1,1)
    y_hist = y_hist.reshape(-1,1)

    hist = np.hstack((t_hist, y_hist))
    df = pd.DataFrame(hist, columns=['t', 'y'])
    df.to_csv(fpath, index=None)
