# Utility Functions

import logging
import pickle
import torch

def make_logger(name=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)

    logger.addHandler(console)
    return logger

def dump_pickle(address, file):
    with open(address, 'wb') as f:
        pickle.dump(file, f)

def load_pickle(address):
    with open(address, 'rb') as f:
        data = pickle.load(f)
    return data

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, save_path='checkpoint.pt'):
        """
        :param patience: how many times you will wait before earlystopping
        :param save_path: where to save checkpoint
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        self.save_path = save_path

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, val_loss)
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, val_loss)
            self.counter = 0 # reset

    def save_checkpoint(self, model, val_loss):
        if self.verbose:
            print(f"val loss: ({self.val_loss_min:.6f} -> {val_loss:.6f})")
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
