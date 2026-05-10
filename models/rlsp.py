import torch
import torch.nn as nn
import torch.nn.functional as F


class RLSPCell(nn.Module):
    def __init__(
        self,
        scale=4,
        hidden_channels=128,
        filters=128,
        num_layers=7,
    ):
        super().__init__()

        self.scale = scale
        self.hidden_channels = hidden_channels

        # xt-1, xt, xt+1 - 3 RGB frames
        input_channels = 3 * 3
        # Previous HR frame - 1 RGB upscaled frame
        feedback_channels = 3 * scale**2
        total_in = input_channels + feedback_channels + hidden_channels

        layers = []

        # First conv
        layers.append(nn.Conv2d(total_in, filters, 3, padding=1))
        layers.append(nn.ReLU(inplace=True))

        # Middle convs
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(filters, filters, 3, padding=1))
            layers.append(nn.ReLU(inplace=True))

        self.body = nn.Sequential(*layers)

        # Hidden state
        self.hidden_conv = nn.Sequential(nn.Conv2d(filters, hidden_channels, 3, padding=1), nn.ReLU(inplace=True))
        # Residual prediction in LR space
        self.residual_conv = nn.Conv2d(filters, 3 * scale * scale, 3, padding=1)

    def forward(
        self,
        x_prev,
        x_curr,
        x_next,
        h_prev,
        y_prev,
    ):
        """
        Inputs:
            x_prev : [B, 3, H, W]
            x_curr : [B, 3, H, W]
            x_next : [B, 3, H, W]
            h_prev : [B, hidden_dim, H, W]
            y_prev : [B, 3, H * scale, W * scale]

        Returns:
            y : HR RGB output
            h : next hidden state
        """

        # Convert previous HR output to LR space
        y_prev_lr = F.pixel_unshuffle(y_prev, downscale_factor=self.scale)

        # Concatenate inputs
        inp = torch.cat(
            [
                x_prev,
                x_curr,
                x_next,
                h_prev,
                y_prev_lr,
            ],
            dim=1,
        )

        feat = self.body(inp)

        # Next hidden state
        h = self.hidden_conv(feat)

        # Residual in LR space
        residual = self.residual_conv(feat)
        base = x_curr.repeat_interleave(self.scale * self.scale, dim=1)
        out_lr = residual + base
        # LR to HR
        y = F.pixel_shuffle(out_lr, upscale_factor=self.scale)

        return y, h


class RLSP(nn.Module):
    def __init__(
        self,
        scale=4,
        hidden_channels=128,
        filters=128,
        num_layers=7,
    ):
        super().__init__()

        self.scale = scale
        self.hidden_channels = hidden_channels

        self.cell = RLSPCell(
            scale=scale,
            hidden_channels=hidden_channels,
            filters=filters,
            num_layers=num_layers,
        )

    def forward(self, x):
        """
        Inputs:
            x: [B, T, 3, H, W]

        Returns:
            y: [B, T, 3, H * scale, W * scale]
        """

        device = x.device

        b, t, c, h, w = x.shape

        # Initialize hidden state
        h_state = torch.zeros(
            b,
            self.hidden_channels,
            h,
            w,
            device=device,
        )

        # Initialize previous output
        y_prev = F.interpolate(
            x[:, 0],
            scale_factor=self.scale,
            mode="bilinear",
            align_corners=False,
        )

        outputs = []

        for i in range(t):
            x_prev = x[:, max(i - 1, 0)]
            x_curr = x[:, i]
            x_next = x[:, min(i + 1, t - 1)]

            y, h_state = self.cell(
                x_prev,
                x_curr,
                x_next,
                h_state,
                y_prev,
            )
            outputs.append(y)

            y_prev = y

        outputs = torch.stack(outputs, dim=1)
        return outputs
