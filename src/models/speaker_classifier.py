import json
import os
from ..base import Classifier
from torch import nn
import torch
import logging
import torch.nn.functional as F
from torchaudio.models import Conformer

logger = logging.getLogger(__name__)


class SelfAttentivePooling(nn.Module):
    def __init__(self, dim):
        """SAP
        Paper: Self-Attentive Speaker Embeddings for Text-Independent Speaker Verification
        Link： https://danielpovey.com/files/2018_interspeech_xvector_attention.pdf
        Args:
            dim (pair): the size of attention weights
        """
        super().__init__()
        self.sap_linear = nn.Linear(dim, dim)
        self.attention = nn.Parameter(torch.FloatTensor(dim, 1))

    def forward(self, x):
        """Computes Self-Attentive Pooling Module
        Args:
            x (torch.Tensor): Input tensor (#batch, dim, frames).
        Returns:
            torch.Tensor: Output tensor (#batch, dim)
        """
        h = torch.tanh(self.sap_linear(x))
        w = torch.matmul(h, self.attention).squeeze(dim=2)
        w = F.softmax(w, dim=1).view(x.size(0), x.size(1), 1)
        x = torch.sum(x * w, dim=1)
        return x


class AMSoftmax(nn.Module):
    """
    Additive Margin Softmax as proposed in:
    https://arxiv.org/pdf/1801.05599.pdf
    """

    def __init__(self, in_features, n_classes, s=30, m=0.4):
        super(AMSoftmax, self).__init__()
        self.linear = nn.Linear(in_features, n_classes, bias=False)
        self.m = m
        self.s = s

    def _am_logsumexp(self, logits):
        max_x = torch.max(logits, dim=-1)[0].unsqueeze(-1)
        term1 = (self.s * (logits - (max_x + self.m))).exp()
        term2 = (self.s * (logits - max_x)).exp().sum(-1).unsqueeze(-1) - (
            self.s * (logits - max_x)
        ).exp()
        return self.s * max_x + (term1 + term2).log()

    def forward(self, *inputs):
        x_vector = F.normalize(inputs[0], p=2, dim=-1)
        self.linear.weight.data = F.normalize(self.linear.weight.data, p=2, dim=-1)
        logits = self.linear(x_vector)
        scaled_logits = (logits - self.m) * self.s
        return scaled_logits - self._am_logsumexp(logits)


class SpeakerClassifier(Classifier):
    def __init__(self, d_model, lr, dropout, weight_decay):
        super().__init__(lr)
        self.save_hyperparameters()
        self.linear = nn.LazyLinear(d_model)
        self.conformer = Conformer(
            input_dim=d_model,
            num_heads=4,
            ffn_dim=128,
            num_layers=4,
            depthwise_conv_kernel_size=31,
            dropout=dropout,
        )
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model, 2, dim_feedforward=256, dropout=dropout, activation="gelu"
        )
        self.pooling = SelfAttentivePooling(d_model)
        self.am_softmax = AMSoftmax(d_model, n_classes=600)  # 600 speakers

    def forward(self, inputs, lengths):
        inputs = self.linear(inputs)

        # inputs: (batch size, length, d_model).
        # lengths: (batch size,)
        inputs, _ = self.conformer(inputs, lengths)

        inputs = self.pooling(inputs)
        return self.am_softmax(inputs)  # (batch, num_speaker)

    def step(self, batch):
        mels, labels, mels_lengths = batch
        logits = self(mels, mels_lengths)
        loss = self.loss(logits, labels)
        self.metrics.add("loss", loss.item(), "acc", self.accuracy(logits, labels))
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def inference(self, dataloader, data_dir):
        logger.info(f"Inference the test set.")
        mapping = json.load(open(os.path.join(data_dir, "mapping.json")))
        self.cuda()
        self.eval()
        results = []
        for feat_paths, mels, input_lengths in dataloader:
            with torch.no_grad():
                mels, input_lengths = mels.cuda(), input_lengths.cuda()
                logits = self(mels, input_lengths)
                preds = logits.argmax(1).cpu().numpy()
                for feat_path, pred in zip(feat_paths, preds):
                    results.append([feat_path, mapping["id2speaker"][str(pred)]])
        return results
