import torch
import torch.nn as nn
from torch.nn.modules.activation import MultiheadAttention


class RGBEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super(RGBEncoder, self).__init__()
        self.rgb_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, out_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
    
    def forward(self, x):
        out = self.rgb_encoder(x)
        return out

class DepthEncoder(nn.Module):
    def __init__(self, out_dim=256):
        super(DepthEncoder, self).__init__()
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(128, out_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
    
    def forward(self, x):
        out = self.depth_encoder(x)
        return out

class GlobalTemporal(nn.Module):
    def __init__(self, input_len=10, out_dim=128):
        super(GlobalTemporal, self).__init__()
        self.flatten = nn.Flatten()

        self.rgb_encoder = RGBEncoder(out_dim=256)
        self.depth_encoder = DepthEncoder(out_dim=256)
        # downsample
        self.rgb_downs = nn.Linear(256, 128)
        self.depth_downs = nn.Linear(256, 128)
        self.downs = nn.Linear(input_len*256, out_dim) 
    
    def forward(self, inputs):
        rgb_seq, depth_seq = inputs
        rgb_latent = []
        depth_latent = []
        for t in range(rgb_seq.size(1)):
            out_rgb = self.rgb_encoder(rgb_seq[:, t, :, :, :]).squeeze(-1)
            rgb_latent.append(self.rgb_downs(out_rgb.squeeze(-1)))
            out_depth = self.depth_encoder(depth_seq[:, t, :, :]).squeeze(-1)
            depth_latent.append(self.depth_downs(out_depth.squeeze(-1)))
        rgb_latent = torch.stack(rgb_latent, dim=0).transpose(0, 1)
        depth_latent = torch.stack(depth_latent, dim=0).transpose(0, 1)
        out = self.downs(self.flatten(torch.cat([rgb_latent, depth_latent], -1)))
        return out

class Interactive(nn.Module):
    def __init__(self, input_len=7, out_dim=128):
        super(Interactive, self).__init__()
        self.flatten = nn.Flatten()

        self.rgb_encoder = RGBEncoder(out_dim=256)
        self.depth_encoder = DepthEncoder(out_dim=256)
        # self attention
        self.mha = MultiheadAttention(256, 4, batch_first=True)
        # downsample
        self.rgb_downs = nn.Linear(256, 128)
        self.depth_downs = nn.Linear(256, 128)
        self.downs = nn.Linear(input_len*256, out_dim) 
    
    def forward(self, inputs):
        rgb_seq, depth_seq = inputs
        rgb_latent = []
        depth_latent = []
        for t in range(rgb_seq.size(1)):
            out_rgb = self.rgb_encoder(rgb_seq[:, t, :, :, :]).squeeze(-1)
            rgb_latent.append(self.rgb_downs(out_rgb.squeeze(-1)))
            out_depth = self.depth_encoder(depth_seq[:, t, :, :]).squeeze(-1)
            depth_latent.append(self.depth_downs(out_depth.squeeze(-1)))
        rgb_latent = torch.stack(rgb_latent, dim=0).transpose(0, 1)
        depth_latent = torch.stack(depth_latent, dim=0).transpose(0, 1)
        interactive_latent = torch.cat([rgb_latent, depth_latent], -1)
        mha_out, _ = self.mha(interactive_latent, interactive_latent, interactive_latent)
        out = self.downs(self.flatten(mha_out))
        return interactive_latent, out

class StateRetrieval(nn.Module):
    def __init__(self, input_len=7, out_dim=128):
        super(StateRetrieval, self).__init__()
        self.flatten = nn.Flatten()

        self.local_rgb = RGBEncoder(out_dim=256)
        self.local_depth = DepthEncoder(out_dim=256)
        # downsample
        self.rgb_downs = nn.Linear(256, 128)
        self.depth_downs = nn.Linear(256, 128)
        self.local_downs = nn.Linear(256, 32)
        # cross attention
        self.cross_mha = MultiheadAttention(260, 4, batch_first=True)
        self.mha_downs = nn.Linear(260, 32)

    
    def forward(self, inputs):
        rgb, depth, ee, interaction_latent, interaction_ee = inputs

        # encode local view
        out_rgb = self.local_rgb(rgb).squeeze(-1)
        rgb_latent = self.rgb_downs((out_rgb.squeeze(-1)))
        out_depth = self.local_depth(depth).squeeze(-1)
        depth_latent = self.depth_downs(out_depth.squeeze(-1))
        local_latent =  torch.cat([rgb_latent, depth_latent], -1)
        local_input = self.local_downs(local_latent)

        # concat observation and ee pose
        local_state = torch.cat([local_latent, ee], -1).unsqueeze(1)
        interaction_state = torch.cat([interaction_latent, interaction_ee], -1)

        # cross attention
        srm_out, mha_cross_weight = self.cross_mha(local_state, interaction_state, interaction_state)
        srm_out = self.mha_downs(self.flatten(srm_out))

        return local_input, srm_out

class SCONE(nn.Module):
    def __init__(self):
        super(SCONE, self).__init__()
        self.global_encoder = GlobalTemporal()
        self.interactive_encoder = Interactive()
        self.SRM = StateRetrieval()
        self.bc = nn.Sequential(
            nn.Linear(128+32+32+128, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
        )
    
    def forward(self, inputs):
        ob_rgb, ob_depth, local_rgb, local_depth, ob_ee, interact_rgb, interact_depth, interact_ee = inputs
        global_input = self.global_encoder((ob_rgb, ob_depth))
        interactive_latent, interactive_input = self.interactive_encoder((interact_rgb, interact_depth))
        local_input, srm_out = self.SRM((local_rgb, local_depth, ob_ee, interactive_latent, interact_ee))
        bc_input = torch.cat([global_input, local_input, srm_out, interactive_input], -1)
        action = self.bc(bc_input)
        return action
