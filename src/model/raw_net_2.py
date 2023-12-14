import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.blocks import SincConv_fast, ResBlock

class RawNet2(nn.Module):
    def __init__(
        self,
        sinc_filter_length = 1024,
        sinc_channels = 128,
        min_low_hz = 0,
        min_band_hz = 0,
        abs_after_sinc = True,
        block1_channels = 20,
        block1_num_layers = 2,
        block2_channels = 128,
        block2_num_layers = 4,
        gru_hidden = 1024,
        gru_num_layers = 3,
        fc_hidden = 1024
    ):
        super().__init__()
        self.sinc = SincConv_fast(
            in_channels = 1,
            out_channels = sinc_channels,
            kernel_size = sinc_filter_length,
            min_low_hz = min_low_hz,
            min_band_hz = min_band_hz,
        )

        self.abs_after_sinc = abs_after_sinc

        self.sinc_head = nn.MaxPool1d(kernel_size=3)
        

        block1_layers = [
            ResBlock(
                in_channels=block1_channels,
                out_channels=block1_channels,
            ) 
            for _ in range(block1_num_layers)
        ]
        block1_layers[0] = ResBlock(
            in_channels=sinc_channels,
            out_channels=block1_channels,
        )
        self.block1 = nn.Sequential(*block1_layers)

        block2_layers = [
            ResBlock(
                in_channels=block2_channels,
                out_channels=block2_channels,
            ) 
            for _ in range(block2_num_layers)
        ]
        block2_layers[0] = ResBlock(
            in_channels=block1_channels,
            out_channels=block2_channels,
        )
        self.block2 = nn.Sequential(*block2_layers)


        self.pre_gru = nn.Sequential(
            nn.BatchNorm1d(num_features=block2_channels),
            nn.LeakyReLU()
        )

        self.gru = nn.GRU(
            input_size=block2_channels,
            hidden_size=gru_hidden,
            num_layers=gru_num_layers,
            batch_first=True,
        )

        # two linear layers one after the other is weird
        # but it is like that in the article
        self.head = nn.Sequential(
            nn.Linear(in_features=gru_hidden, out_features=fc_hidden),
            nn.Linear(in_features=fc_hidden, out_features=2),
        )



    def forward(self, audio, **batch):
        x = self.sinc(audio)
        if self.abs_after_sinc:
            x = torch.abs(x)
        x = self.sinc_head(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.pre_gru(x)
        x = x.permute(0, 2, 1)
        x, _ = self.gru(x)
        x = x[:, -1, :]
        x = self.head(x)
        if not self.training:
            x = F.softmax(x, dim=-1)

        return {"prediction": x}
