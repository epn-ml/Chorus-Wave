#!/usr/bin/env python

"""models.py contains Pytorch models"""

__author__      = "Sahib Julka <sahib.julka@uni-passau.de>"
__copyright__   = "GPL"


import torch.nn as nn
import torch
import pytorch_lightning as pl
import segmentation_models_pytorch as smp



# %%
# adapted from pytorch-segmentation models repo
class Segmentor(pl.LightningModule):
    def __init__(self, arch, encoder_name, in_channels, out_classes, **kwargs):
        """
        Initializes the Segmentor model.

        Args:
            arch (str): Architecture name.
            encoder_name (str): Pre-trained encoder name.
            in_channels (int): Number of input channels.
            out_classes (int): Number of output classes.
            **kwargs: Additional arguments for the model.
        """
        super().__init__()
        self.model = smp.create_model(
            arch, encoder_name=encoder_name, in_channels=in_channels, classes=out_classes, **kwargs
        )

        # Preprocessing parameters for image normalization
        params = smp.encoders.get_preprocessing_params(encoder_name)
        self.register_buffer("std", torch.tensor(params["std"]).view(1, 3, 1, 1))
        self.register_buffer("mean", torch.tensor(params["mean"]).view(1, 3, 1, 1))

        # Loss function for image segmentation (Dice Loss)
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def forward(self, image):
        """
        Forward pass of the model.

        Args:
            image (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Predicted mask.
        """
        # Normalize the image
        image = (image - self.mean) / self.std
        mask = self.model(image)
        return mask

    def shared_step(self, batch, stage):
        """
        Shared step for training, validation, and testing.

        Args:
            batch (dict): Batch of input data.
            stage (str): Current stage (train, valid, or test).

        Returns:
            dict: Dictionary containing the loss and evaluation metrics.
        """
        image = batch["image"]
        mask = batch["mask"]

        # Normalize the image
        image = (image - self.mean) / self.std

        # Perform forward pass
        logits_mask = self.forward(image)

        # Calculate the loss
        loss = self.loss_fn(logits_mask, mask)

        # Compute predicted mask and evaluation metrics
        prob_mask = logits_mask.sigmoid()
        pred_mask = (prob_mask > 0.5).float()
        tp, fp, fn, tn = smp.metrics.get_stats(pred_mask.long(), mask.long(), mode="binary")

        return {
            "loss": loss,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        }

    def shared_epoch_end(self, outputs, stage):
        """
        Shared epoch end function for training, validation, and testing.

        Args:
            outputs (list): List of output dictionaries from each step.
            stage (str): Current stage (train, valid, or test).
        """
        # Aggregate step metrics
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])

        # Compute per image IoU and dataset IoU
        per_image_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise")
        dataset_iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro")
        dataset_f1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction="micro")

        # Log the metrics
        metrics = {
            f"{stage}_per_image_iou": per_image_iou,
            f"{stage}_dataset_iou": dataset_iou,
            f"{stage}_f1_score": dataset_f1,
        }
        self.log_dict(metrics, prog_bar=True)

    def training_step(self, batch, batch_idx):
        """
        Training step.

        Args:
            batch (dict): Batch of input data.
            batch_idx (int): Index of the current batch.

        Returns:
            torch.Tensor: Loss value.
        """
        output = self.shared_step(batch, "train")
        self.log("train_loss", output["loss"])
        return output["loss"]

    def training_epoch_end(self, outputs):
        """
        Training epoch end function.

        Args:
            outputs (list): List of output dictionaries from each step.
        """
        self.shared_epoch_end(outputs, "train")

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Args:
            batch (dict): Batch of input data.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: Dictionary containing the loss and evaluation metrics.
        """
        return self.shared_step(batch, "valid")

    def validation_epoch_end(self, outputs):
        """
        Validation epoch end function.

        Args:
            outputs (list): List of output dictionaries from each step.
        """
        self.shared_epoch_end(outputs, "valid")

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Args:
            batch (dict): Batch of input data.
            batch_idx (int): Index of the current batch.

        Returns:
            dict: Dictionary containing the loss and evaluation metrics.
        """
        return self.shared_step(batch, "test")

    def test_epoch_end(self, outputs):
        """
        Test epoch end function.

        Args:
            outputs (list): List of output dictionaries from each step.
        """
        self.shared_epoch_end(outputs, "test")

    def configure_optimizers(self):
        """
        Configure the optimizer.

        Returns:
            torch.optim.Optimizer: Optimizer for training the model.
        """
        return torch.optim.Adam(self.parameters(), lr=0.0001)






    
class Net:
    def __init__(self, net, params, device):
        """
        Wrapper class for the neural network model.

        Args:
            net: Neural network model.
            params (dict): Parameters for training and testing.
            device: Device to use for training and testing.
        """
        self.net = net
        self.params = params
        self.device = device
        self.loss_fn = smp.losses.DiceLoss(smp.losses.BINARY_MODE, from_logits=True)

    def train(self, data):
        """
        Train the neural network model.

        Args:
            data: Training data.

        Returns:
            None
        """
        n_epoch = self.params['n_epoch']
        self.clf = self.net.to(self.device)
        self.clf.train()
        optimizer = optim.SGD(self.clf.parameters(), **self.params['optimizer_args'])

        loader = DataLoader(data, shuffle=True, **self.params['train_args'])
        for epoch in tqdm.tqdm(range(1, n_epoch+1), ncols=100):
            for batch_idx, (x, y, idxs) in enumerate(loader):
                x, y = x.to(self.device), y.to(self.device)
                optimizer.zero_grad()
                out = self.clf(x)
                loss = self.loss_fn(out, y)
                loss.backward()
                optimizer.step()

    def predict(self, data):
        """
        Make predictions using the trained model.

        Args:
            data: Test data.

        Returns:
            tuple: Predicted masks and ground truth masks.
        """
        self.clf.eval()
        preds = torch.zeros(len(data), 640 , 1920)
        masks = torch.zeros(len(data), 640, 1920)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                preds[idxs] = out.squeeze().cpu()
                masks[idxs] = y.squeeze().cpu()
        return preds, masks

    def predict_prob(self, data):
        """
        Make probability predictions using the trained model.

        Args:
            data: Test data.

        Returns:
            torch.Tensor: Predicted probabilities.
        """
        self.clf.eval()
        probs = torch.zeros(len(data), 640, 1920)
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        with torch.no_grad():
            for x, y, idxs in loader:
                x, y = x.to(self.device), y.to(self.device)
                out = self.clf(x)
                prob = F.sigmoid(out)
                probs[idxs] = prob.squeeze().cpu()
        return probs

    def predict_prob_dropout(self, data, n_drop=10):
        """
        Make probability predictions with dropout using the trained model.

        Args:
            data: Test data.
            n_drop (int): Number of dropout samples.

        Returns:
            torch.Tensor: Predicted probabilities with dropout.
        """
        self.clf.train()
        probs = torch.zeros([len(data), 640 * 1920])
        loader = DataLoader(data, shuffle=False, **self.params['test_args'])
        for i in range(n_drop):
            with torch.no_grad():



class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Upsample layers
        self.upsample1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.upsample3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        # Output layers
        self.output = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
            
        )

    def forward(self, x):
        x = self.upsample1(x)
        x = self.upsample2(x)
        x = self.upsample3(x)
        x = self.output(x)
        # Upsample output to (N, 3, 1024, 1024)
        x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        return x
