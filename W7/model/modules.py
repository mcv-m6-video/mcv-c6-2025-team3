"""
File containing the different modules related to the model: T-DEED.
"""

#Standard imports
import abc
import torch
import torch.nn as nn
import math

#Local imports

class ABCModel:

    @abc.abstractmethod
    def get_optimizer(self, opt_args):
        raise NotImplementedError()

    @abc.abstractmethod
    def epoch(self, loader, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def predict(self, seq):
        raise NotImplementedError()

    @abc.abstractmethod
    def state_dict(self):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, state_dict):
        raise NotImplementedError()

class BaseRGBModel(ABCModel):

    def get_optimizer(self, opt_args):
        return torch.optim.AdamW(self._get_params(), **opt_args), \
            torch.cuda.amp.GradScaler() if self.device == 'cuda' else None

    """ Assume there is a self._model """

    def _get_params(self):
        return list(self._model.parameters())

    def state_dict(self):
        if isinstance(self._model, nn.DataParallel):
            return self._model.module.state_dict()
        return self._model.state_dict()

    def load(self, state_dict):
        if isinstance(self._model, nn.DataParallel):
            self._model.module.load_state_dict(state_dict)
        else:
            self._model.load_state_dict(state_dict)    

class FCLayers(nn.Module):

    def __init__(self, feat_dim, num_classes):
        super().__init__()
        self._fc_out = nn.Linear(feat_dim, num_classes)
        self.dropout = nn.Dropout()

    def forward(self, x):
        if len(x.shape) == 3:
            b, t, d = x.shape
            x = x.reshape(b * t, d)
            return self._fc_out(self.dropout(x)).reshape(b, t, -1)
        elif len(x.shape) == 2:
            return self._fc_out(self.dropout(x))

def step(optimizer, scaler, loss, first_step, lr_scheduler=None):
    if scaler is None:
        loss.backward()
    else:
        scaler.scale(loss).backward()

    if scaler is None:
        optimizer.step()
    else:
        scaler.step(optimizer)
        scaler.update()

    if lr_scheduler is not None and not first_step:
        lr_scheduler.step()

    optimizer.zero_grad()

def compute_class_weights(dataset, num_classes):
    class_counts = torch.zeros(num_classes, dtype=torch.long)
    total_count = 0
    
    for item in dataset:
        labels = item['label']  
        labels = labels.view(-1)
        class_counts += torch.bincount(labels, minlength=num_classes)
        total_count += labels.numel()

    print("Class counts:", class_counts)
    print("Total frames:", total_count)

    freq = class_counts.float() / total_count

    print("Frequencies:", freq)

    freq_nonzero = freq[freq>0]
    if len(freq_nonzero) == 0:
        print("¡Error, no hay clases con freq>0!")
        return torch.ones(num_classes)

    median_freq = freq_nonzero.median()
    
    print("Median frequency:", median_freq.item())

    w = median_freq / freq
    #w = torch.log1p(w)
    w = torch.sqrt(w)
    print("Class Weights1:", w)

    w = w / w.mean()
    print("Class Weights2:", w)

    w = w.clamp(min=(max(w)*0.005), max=1.0)
    print("Class Weights3:", w)

    strong = torch.tensor([0.5] + [5.0] * (num_classes-1)).to(w.device)
    w = w + strong
    
    print("Class Weights4:", w)

    return w