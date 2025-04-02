"""
File containing the main model.
"""

#Standard imports
import torch
from torch import nn
import timm
import torchvision.transforms as T
from contextlib import nullcontext
from tqdm import tqdm
import torch.nn.functional as F
import sys
from sklearn.metrics import average_precision_score
from ptflops import get_model_complexity_info


#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class RGBClassifier(nn.Module):
    def __init__(self, args = None):
        super().__init__()
        self._feature_arch = args.feature_arch

        if self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
            features = timm.create_model({
                'rny002': 'regnety_002',
                'rny004': 'regnety_004',
                'rny008': 'regnety_008',
            }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
            feat_dim = features.head.fc.in_features

            # Remove final classification layer
            features.head.fc = nn.Identity()
            self._d = feat_dim

        else:
            raise NotImplementedError(args._feature_arch)

        self._features = features

        # MLP for classification
        self._fc = FCLayers(self._d, args.num_classes)

        #Augmentations and crop
        self.augmentation = T.Compose([
            T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
            T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
            T.RandomApply([T.GaussianBlur(5)], p = 0.25),
            T.RandomHorizontalFlip(),
        ])

        #Standarization
        self.standarization = T.Compose([
            T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
        ])

        
        self.attn_pool = nn.Sequential(
            nn.LayerNorm(self._d),
            nn.Linear(self._d, 128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128, 1),
            nn.Softmax(dim=1)
        )
        
        
    def forward(self, x):
        x = self.normalize(x) #Normalize to 0-1
        batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

        if self.training:
            x = self.augment(x) #augmentation per-batch

        x = self.standarize(x) #standarization imagenet stats
                    
        im_feat = self._features(
            x.view(-1, channels, height, width)
        ).reshape(batch_size, clip_len, self._d) #B, T, D

        #Max pooling
        #im_feat = torch.max(im_feat, dim=1)[0] #B, D
        #im_feat = torch.mean(im_feat, dim=1)

        #im_feat = torch.cat([im_max, im_avg], dim=1)

        weights = self.attn_pool(im_feat) # (B, T, 1)
        im_feat = (weights * im_feat).sum(dim=1) #B, D

        #MLP
        im_feat = self._fc(im_feat) #B, num_classes

        return im_feat 
    
    def normalize(self, x):
        return x / 255.
    
    def augment(self, x):
        for i in range(x.shape[0]):
            x[i] = self.augmentation(x[i])
        return x

    def standarize(self, x):
        for i in range(x.shape[0]):
            x[i] = self.standarization(x[i])
        return x

    def print_stats(self):

        print('Model params:',
            sum(p.numel() for p in self.parameters()))

        def input_constructor(input_res):
            batch = torch.randn(1, 1, *input_res)  # (1, 1, 3, 398, 224)
            return {'x': batch}

        macs, _ = get_model_complexity_info(self, (3, 398, 224), input_constructor=input_constructor,
        as_strings=True, print_per_layer_stat=False, verbose=False
        )
        
        print('Model MACs:', macs)


class Model(BaseRGBModel):
    def __init__(self, args=None):
        self.device = "cpu"
        if torch.cuda.is_available() and ("device" in args) and (args.device == "cuda"):
            self.device = "cuda"

        self._model = RGBClassifier(args)
        self._model.print_stats()
        self._args = args

        self._model.to(self.device)
        self._num_classes = args.num_classes

    def epoch(self, loader, optimizer=None, scaler=None, lr_scheduler=None, pos_weight=None):

        disable_tqdm = not sys.stdout.isatty()

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.

        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader, disable=disable_tqdm)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).float()

                with torch.cuda.amp.autocast():
                    pred = self._model(frame)
                    loss = F.binary_cross_entropy_with_logits(
                        pred, label, pos_weight=pos_weight)

                if not inference:
                    step(optimizer, scaler, loss,
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader) # Avg loss

    def predict(self, seq):

        if not isinstance(seq, torch.Tensor):
            seq = torch.FloatTensor(seq)
        if len(seq.shape) == 4: # (L, C, H, W)
            seq = seq.unsqueeze(0)
        if seq.device != self.device:
            seq = seq.to(self.device)
        seq = seq.float()

        self._model.eval()
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                pred = self._model(seq)

            # apply sigmoid
            pred = torch.sigmoid(pred)
            
            return pred.cpu().numpy()
