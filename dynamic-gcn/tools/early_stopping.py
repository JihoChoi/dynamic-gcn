# Reference
# https://github.com/Bjarten/early-stopping-pytorch/blob/master/pytorchtools.py


import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, model_path='checkpoint.pt', delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.model_path = model_path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, model_name, dataset_name, eval_result):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.eval_result = eval_result
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("Early Stopping - Update Counter: {}/{}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
                print()
                print("BEST: score {:.4f} | accuracy {:.4f}".format(
                    self.best_score * -1, self.eval_result['acc_all'])
                )
                print("eval_result:", self.eval_result)
        else:  # Update Best Score
            self.counter = 0
            print("Early Stopping - Update Score: {}/{}".format(self.counter, self.patience))
            self.best_score = score
            self.eval_result = eval_result
            self.save_checkpoint(val_loss, model)

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print("\t Validation loss decreased ({:.6f} -> {:.6f})".format(
                self.val_loss_min, val_loss)
            )
            print("Saving model ...")
        torch.save(model.state_dict(), self.model_path)
        self.val_loss_min = val_loss