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
from ptflops import get_model_complexity_info


#Local imports
from model.modules import BaseRGBModel, FCLayers, step

class RGBClassifier(nn.Module):

        def __init__(self, args = None):
            super().__init__()
            self._feature_arch = args.feature_arch

            self._use_bypass = True

            if args.feature_arch == "swin":
                features = timm.create_model("swin_tiny_patch4_window7_224", pretrained=True)
                feat_dim = features.head.in_features
                features.head = nn.Identity()
                self._d = feat_dim

            elif args.feature_arch == "effb0":
                features = timm.create_model("efficientnet_b0", pretrained=True)
                feat_dim = features.classifier.in_features
                features.classifier = nn.Identity()
                self._d = feat_dim

            elif self._feature_arch.startswith(('rny002', 'rny004', 'rny008')):
                features = timm.create_model({
                    'rny002': 'regnety_002',
                    'rny004': 'regnety_004',
                    'rny008': 'regnety_008',
                }[self._feature_arch.rsplit('_', 1)[0]], pretrained=True)
                feat_dim = features.head.fc.in_features

                # Remove final classification layer
                features.head.fc = nn.Identity()
                self._d = feat_dim  #features dim

            else:
                raise NotImplementedError(args._feature_arch)

            self._features = features

            #transformer    
            """
            self._len = args.clip_len #n of frames
            self._positional_encoding = nn.Parameter(torch.randn(1, self._len, self._d))

            self._encoder_layer = nn.TransformerEncoderLayer(
                d_model=self._d, #dim of features
                nhead=8, #heads
                dim_feedforward=self._d * 4, #intern mlp
                dropout=0.1,
                batch_first=True 
                )

            self._transformer = nn.TransformerEncoder(
                self._encoder_layer,
                num_layers= 4 #layers
                )
            """

            #tcn
            self._tcn = nn.Sequential(
                nn.Conv1d(self._d, self._d, kernel_size=5, padding=2, dilation=1),
                nn.ReLU(),
                nn.Conv1d(self._d, self._d, kernel_size=5, padding=4, dilation=2),
                nn.ReLU(),
                nn.Conv1d(self._d, self._d, kernel_size=5, padding=8, dilation=4),
                nn.ReLU(),
                )


            # MLP for classification
            self._fc = FCLayers(self._d, args.num_classes+1) # +1 for background class (we now perform per-frame classification with softmax, therefore we have the extra background class)

            """
            #Augmentations and crop
            self.augmentation = T.Compose([
                T.RandomApply([T.ColorJitter(hue = 0.2)], p = 0.25),
                T.RandomApply([T.ColorJitter(saturation = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(brightness = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.ColorJitter(contrast = (0.7, 1.2))], p = 0.25),
                T.RandomApply([T.GaussianBlur(5)], p = 0.25),
                T.RandomHorizontalFlip(),
            ])
            """

            #Standarization
            self.standarization = T.Compose([
                T.Normalize(mean = (0.485, 0.456, 0.406), std = (0.229, 0.224, 0.225)) #Imagenet mean and std
            ])

        def forward(self, x):
            x = self.normalize(x) #Normalize to 0-1
            batch_size, clip_len, channels, height, width = x.shape #B, T, C, H, W

            if self.training:
                x = self.augment(x) #augmentation per-batch

            x = self.standarize(x) #standarization imagenet stats
                        
            im_feat = self._features(
                x.view(-1, channels, height, width)
            ).reshape(batch_size, clip_len, self._d) #B, T, D
            """
            #positional encoding
            pos_enc = self._positional_encoding[:, :clip_len, :]   # [1, T, d]
            im_feat_pos = im_feat + pos_enc

            #apply transformer
            im_feat_result = self._transformer(im_feat_pos) # [B, T, d]

            #skip connection
            if self._use_bypass:
                im_feat_result += im_feat
            """
            im_feat = im_feat.transpose(1, 2)
            im_feat = self._tcn(im_feat)
            im_feat_result = im_feat.transpose(1, 2)

            #MLP
            logits = self._fc(im_feat_result) #B, T, num_classes+1

            return logits 
        
        def normalize(self, x):
            return x / 255.

        """
        def augment(self, x):
            for i in range(x.shape[0]):
                x[i] = self.augmentation(x[i])
            return x
        """

        def augment(self, x):
            B, clip_len, C, H, W = x.shape
            x_aug = torch.zeros_like(x)

            for i in range(B):  
                clip = x[i]  # [T, C, H, W]

                #parameters
                do_flip = torch.rand(1).item() < 0.5
                do_blur = torch.rand(1).item() < 0.25
                do_jitter_hue = torch.rand(1).item() < 0.25
                do_jitter_sat = torch.rand(1).item() < 0.25
                do_jitter_bright = torch.rand(1).item() < 0.25
                do_jitter_contrast = torch.rand(1).item() < 0.25

                hue_factor = (torch.rand(1).item() - 0.5) * 0.4 if do_jitter_hue else 0.0
                sat_factor = 0.7 + torch.rand(1).item() * 0.5 if do_jitter_sat else 1.0
                bright_factor = 0.7 + torch.rand(1).item() * 0.5 if do_jitter_bright else 1.0
                contrast_factor = 0.7 + torch.rand(1).item() * 0.5 if do_jitter_contrast else 1.0

                for t in range(clip_len):
                    frame = clip[t]

                    #ColorJitter
                    if do_jitter_hue:
                        frame = T.functional.adjust_hue(frame, hue_factor)
                    if do_jitter_sat:
                        frame = T.functional.adjust_saturation(frame, sat_factor)
                    if do_jitter_bright:
                        frame = T.functional.adjust_brightness(frame, bright_factor)
                    if do_jitter_contrast:
                        frame = T.functional.adjust_contrast(frame, contrast_factor)

                    #Gaussian blur
                    if do_blur:
                        frame = T.functional.gaussian_blur(frame, kernel_size=5)

                    # flip
                    if do_flip:
                        frame = T.functional.hflip(frame)

                    x_aug[i, t] = frame
            return x_aug

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

    def epoch(self, loader, weights=None, epoch=None, optimizer=None, scaler=None,  lr_scheduler=None):

        if optimizer is None:
            inference = True
            self._model.eval()
        else:
            inference = False
            optimizer.zero_grad()
            self._model.train()

        epoch_loss = 0.
        with torch.no_grad() if optimizer is None else nullcontext():
            for batch_idx, batch in enumerate(tqdm(loader)):
                frame = batch['frame'].to(self.device).float()
                label = batch['label']
                label = label.to(self.device).long()

                with torch.cuda.amp.autocast():
                    if not inference:
                        if epoch == 5:
                            self._model._use_bypass = False

                    pred = self._model(frame)
                    pred = pred.view(-1, self._num_classes + 1) # B*T, num_classes
                    label = label.view(-1) # B*T
                    """
                    loss = F.cross_entropy(
                            pred, label, reduction='mean', weight = weights)
                    """
                    gamma = 1
                    ce_loss = F.cross_entropy(pred, label, reduction='none')
                    pt = torch.exp(-ce_loss)
                    alpha_t = weights[label]
                    loss = (alpha_t * (1 - pt) ** gamma * ce_loss).mean()
                    
                if not inference:
                    step(optimizer, scaler, loss, (epoch== 0 and batch_idx==0), 
                        lr_scheduler=lr_scheduler)

                epoch_loss += loss.detach().item()

        return epoch_loss / len(loader)     # Avg loss

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
            pred = torch.softmax(pred, dim=-1)
            
            return pred.cpu().numpy()
