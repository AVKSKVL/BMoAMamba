import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from mamba_ssm.ops.selective_scan_interface import SelectiveScanFn, selective_scan_fn


device = torch.device("cuda:1")


class upms(nn.Module):
    def __init__(self):
        super(upms, self).__init__()

    def forward(self, x):

        return F.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)

class R1(nn.Module):
    def __init__(self):
        super(R1, self).__init__()
        self.conv = nn.Conv2d(in_channels=1, out_channels=4, kernel_size=1)

    def forward(self, x):

        return self.conv(x)

class Classifier(nn.Module):
    def __init__(self, Classes):
        super().__init__()
        self.norm1 = nn.LayerNorm(1024)
        self.act = nn.GELU()
        self.classifier_1 = nn.Linear(4 * 64 * 64, 1024)
        self.classifier_2 = nn.Linear(1024, 4)

    def forward(self, x_in):
        x = self.classifier_1(x_in)
        return self.classifier_2(x)

class Permute(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.args = args

    def forward(self, x: torch.Tensor):
        return x.permute(*self.args)

class mamba_init:
    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):
        # S4D real initialization
        A = torch.arange(1, d_state + 1, dtype=torch.float32, device=device).view(1, -1).repeat(d_inner, 1).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 0:
            A_log = A_log[None].repeat(copies, 1, 1).contiguous()
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = D[None].repeat(copies, 1).contiguous()
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    @classmethod
    def init_dt_A_D(cls, d_state, dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, k_group=4):
        # dt proj ============================
        dt_projs = [
            cls.dt_init(dt_rank, d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor)
            for _ in range(k_group)
        ]
        dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in dt_projs], dim=0))  # (K, inner, rank)
        dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in dt_projs], dim=0))  # (K, inner)
        del dt_projs

        # A, D =======================================
        A_logs = cls.A_log_init(d_state, d_inner, copies=k_group, merge=True)  # (K * D, N)
        Ds = cls.D_init(d_inner, copies=k_group, merge=True)  # (K * D)
        return A_logs, Ds, dt_projs_weight, dt_projs_bias

class MAMBA(nn.Module):
    def __init__(self, d_model, d_state=24, d_conv=3, expand=1, dt_rank='auto', bias=False, device=None, dtype=None, ):
        super(MAMBA, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.d_model * self.expand)
        self.d_conv = d_conv
        # self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == 'auto' else dt_rank
        self.dt_rank = 24
        # new
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=False)
        self.act = nn.SiLU()
        self.conv2d = nn.Conv2d(in_channels=4*expand, out_channels=4*expand, groups=4*expand, bias=True,
                                kernel_size=d_conv, padding=(d_conv - 1) // 2)

        self.x_proj_1 = [
            nn.Linear(int(self.d_model / 4), 18, bias=False)
            for _ in range(4)
        ]
        self.x_proj_weight_1 = nn.Parameter(torch.stack([t.weight for t in self.x_proj_1], dim=0))
        del self.x_proj_1

        self.x_proj_2 = [
            nn.Linear(int(self.d_model / 4), 18, bias=False)
            for _ in range(4)
        ]
        self.x_proj_weight_2 = nn.Parameter(torch.stack([t.weight for t in self.x_proj_2], dim=0))
        del self.x_proj_2

        self.x_proj_3 = [
            nn.Linear(int(self.d_model / 4), 18, bias=False)
            for _ in range(4)
        ]
        self.x_proj_weight_3 = nn.Parameter(torch.stack([t.weight for t in self.x_proj_3], dim=0))
        del self.x_proj_3

        self.x_proj_4 = [
            nn.Linear(int(self.d_model / 4), 18, bias=False)
            for _ in range(4)
        ]
        self.x_proj_weight_4 = nn.Parameter(torch.stack([t.weight for t in self.x_proj_4], dim=0))
        del self.x_proj_4


        self.A_logs1, self.Ds1, self.dt_projs_weight1, self.dt_projs_bias1 = mamba_init.init_dt_A_D(
            6, 6, int(self.d_model / 4), dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
            dt_init_floor=1e-4, k_group=4,
        )
        self.A_logs2, self.Ds2, self.dt_projs_weight2, self.dt_projs_bias2 = mamba_init.init_dt_A_D(
            6, 6, int(self.d_model / 4), dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
            dt_init_floor=1e-4, k_group=4,
        )
        self.A_logs3, self.Ds3, self.dt_projs_weight3, self.dt_projs_bias3 = mamba_init.init_dt_A_D(
            6, 6, int(self.d_model / 4), dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
            dt_init_floor=1e-4, k_group=4,
        )
        self.A_logs4, self.Ds4, self.dt_projs_weight4, self.dt_projs_bias4 = mamba_init.init_dt_A_D(
            6, 6, int(self.d_model / 4), dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1,
            dt_init_floor=1e-4, k_group=4,
        )


        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.patch = nn.Sequential(
            nn.Conv2d(in_channels=4*expand, out_channels=self.d_inner, kernel_size=8, stride=8, bias=True),
            nn.BatchNorm2d(self.d_inner))

    def forward(self, x_in):
        x_ = x_in.clone()
        b, p, q, d = x_.shape
        z = self.in_proj(x_)
        z = self.act(z)
        x_ = x_.reshape(b,8,8,4*self.expand,8,8).permute(0,3,1,4,2,5).reshape(b,4*self.expand,64,64)

        x_ = self.conv2d(x_)
        x_ = self.act(x_)
        x = self.patch(x_)

        x_proj_bias = getattr(self, "x_proj_bias", None)

        R, N = 6, 6

        x_hwwh = torch.stack([x.view(b, self.d_inner, p*q), torch.transpose(x, dim0=2, dim1=3).contiguous().view(b, self.d_inner, p*q)],
                             dim=1).view(b, 2, self.d_inner, p*q)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)

        xs1, xs2, xs3, xs4 = xs.chunk(4, dim=2)

        x_dbl_1 = torch.einsum("b k d l, k c d -> b k c l", xs1, self.x_proj_weight_1)
        x_dbl_2 = torch.einsum("b k d l, k c d -> b k c l", xs2, self.x_proj_weight_2)
        x_dbl_3 = torch.einsum("b k d l, k c d -> b k c l", xs3, self.x_proj_weight_3)
        x_dbl_4 = torch.einsum("b k d l, k c d -> b k c l", xs4, self.x_proj_weight_4)

        dts1, Bs1, Cs1 = torch.split(x_dbl_1, [R, N, N], dim=2)
        dts2, Bs2, Cs2 = torch.split(x_dbl_2, [R, N, N], dim=2)
        dts3, Bs3, Cs3 = torch.split(x_dbl_3, [R, N, N], dim=2)
        dts4, Bs4, Cs4 = torch.split(x_dbl_4, [R, N, N], dim=2)

        dts1 = torch.einsum("b k r l, k d r -> b k d l", dts1, self.dt_projs_weight1)
        dts2 = torch.einsum("b k r l, k d r -> b k d l", dts2, self.dt_projs_weight2)
        dts3 = torch.einsum("b k r l, k d r -> b k d l", dts3, self.dt_projs_weight3)
        dts4 = torch.einsum("b k r l, k d r -> b k d l", dts4, self.dt_projs_weight4)

        xs1 = xs1.contiguous().view(b, (self.d_inner / 4) * 4, p*q)
        xs2 = xs2.contiguous().view(b, (self.d_inner / 4) * 4, p*q)
        xs3 = xs3.contiguous().view(b, (self.d_inner / 4) * 4, p*q)
        xs4 = xs4.contiguous().view(b, (self.d_inner / 4) * 4, p*q)

        dts1 = dts1.contiguous().view(b, -1, p*q)
        dts2 = dts2.contiguous().view(b, -1, p*q)
        dts3 = dts3.contiguous().view(b, -1, p*q)
        dts4 = dts4.contiguous().view(b, -1, p*q)

        Bs1 = Bs1.contiguous()
        Cs1 = Cs1.contiguous()
        As1 = -self.A_logs1.float().exp()
        Ds1 = self.Ds1.float()
        Bs2 = Bs2.contiguous()
        Cs2 = Cs2.contiguous()
        As2 = -self.A_logs2.float().exp()
        Ds2 = self.Ds2.float()
        Bs3 = Bs3.contiguous()
        Cs3 = Cs3.contiguous()
        As3 = -self.A_logs3.float().exp()
        Ds3 = self.Ds3.float()
        Bs4 = Bs4.contiguous()
        Cs4 = Cs4.contiguous()
        As4 = -self.A_logs4.float().exp()
        Ds4 = self.Ds4.float()

        dt_projs_bias1 = self.dt_projs_bias1.float().view(-1)
        dt_projs_bias2 = self.dt_projs_bias2.float().view(-1)
        dt_projs_bias3 = self.dt_projs_bias3.float().view(-1)
        dt_projs_bias4 = self.dt_projs_bias4.float().view(-1)

        out_y1 = []
        out_y2 = []
        out_y3 = []
        out_y4 = []

        for i in range(4):
            yi1 = selective_scan_fn(
                xs1.view(b, 4, -1, p*q)[:, i], dts1.view(b, 4, -1, p*q)[:, i],
                As1.view(4, -1, N)[i], Bs1[:, i].unsqueeze(1), Cs1[:, i].unsqueeze(1), Ds1.view(4, -1)[i],
                delta_bias=dt_projs_bias1.view(4, -1)[i],
                delta_softplus=True,
            ).view(b, -1, p*q)
            out_y1.append(yi1)
        out_y1 = torch.stack(out_y1, dim=1)
        inv_y1 = torch.flip(out_y1[:, 2:4], dims=[-1]).view(b, 2, -1, p*q)
        wh_y1 = torch.transpose(out_y1[:, 1].view(b, -1, q, q), dim0=2, dim1=3).contiguous().view(b, -1, p*q)
        invwh_y1 = torch.transpose(inv_y1[:, 1].view(b, -1, q, q), dim0=2, dim1=3).contiguous().view(b, -1, p*q)
        y1 = out_y1[:, 0] + inv_y1[:, 0] + wh_y1 + invwh_y1
        y1 = y1.transpose(dim0=1, dim1=2).contiguous()

        for i in range(4):
            yi2 = selective_scan_fn(
                xs2.view(b, 4, -1, p*q)[:, i], dts2.view(b, 4, -1, p*q)[:, i],
                As2.view(4, -1, N)[i], Bs2[:, i].unsqueeze(1), Cs2[:, i].unsqueeze(1), Ds2.view(4, -1)[i],
                delta_bias=dt_projs_bias2.view(4, -1)[i],
                delta_softplus=True,
            ).view(b, -1, p*q)
            out_y2.append(yi2)
        out_y2 = torch.stack(out_y2, dim=1)
        inv_y2 = torch.flip(out_y2[:, 2:4], dims=[-1]).view(b, 2, -1, p*q)
        wh_y2 = torch.transpose(out_y2[:, 1].view(b, -1,q,q), dim0=2, dim1=3).contiguous().view(b, -1, p*q)
        invwh_y2 = torch.transpose(inv_y2[:, 1].view(b, -1,q,q), dim0=2, dim1=3).contiguous().view(b, -1, p*q)
        y2 = out_y2[:, 0] + inv_y2[:, 0] + wh_y2 + invwh_y2
        y2 = y2.transpose(dim0=1, dim1=2).contiguous()

        for i in range(4):
            yi3 = selective_scan_fn(
                xs3.view(b, 4, -1, p*q)[:, i], dts3.view(b, 4, -1, p*q)[:, i],
                As3.view(4, -1, N)[i], Bs3[:, i].unsqueeze(1), Cs3[:, i].unsqueeze(1), Ds3.view(4, -1)[i],
                delta_bias=dt_projs_bias3.view(4, -1)[i],
                delta_softplus=True,
            ).view(b, -1, p*q)
            out_y3.append(yi3)
        out_y3 = torch.stack(out_y3, dim=1)
        inv_y3 = torch.flip(out_y3[:, 2:4], dims=[-1]).view(b, 2, -1, p*q)
        wh_y3 = torch.transpose(out_y3[:, 1].view(b, -1, q, q), dim0=2, dim1=3).contiguous().view(b, -1, p*q)
        invwh_y3 = torch.transpose(inv_y3[:, 1].view(b, -1, q, q), dim0=2, dim1=3).contiguous().view(b, -1, p*q)
        y3 = out_y3[:, 0] + inv_y3[:, 0] + wh_y3 + invwh_y3
        y3 = y3.transpose(dim0=1, dim1=2).contiguous()

        for i in range(4):
            yi4 = selective_scan_fn(
                xs4.view(b, 4, -1, p*q)[:, i], dts4.view(b, 4, -1, p*q)[:, i],
                As4.view(4, -1, N)[i], Bs4[:, i].unsqueeze(1), Cs4[:, i].unsqueeze(1), Ds4.view(4, -1)[i],
                delta_bias=dt_projs_bias4.view(4, -1)[i],
                delta_softplus=True,
            ).view(b, -1, p*q)
            out_y4.append(yi4)
        out_y4 = torch.stack(out_y4, dim=1)
        inv_y4 = torch.flip(out_y4[:, 2:4], dims=[-1]).view(b, 2, -1, p*q)
        wh_y4 = torch.transpose(out_y4[:, 1].view(b, -1, q, q), dim0=2, dim1=3).contiguous().view(b, -1, p*q)
        invwh_y4 = torch.transpose(inv_y4[:, 1].view(b, -1, q, q), dim0=2, dim1=3).contiguous().view(b, -1, p*q)
        y4 = out_y4[:, 0] + inv_y4[:, 0] + wh_y4 + invwh_y4
        y4 = y4.transpose(dim0=1, dim1=2).contiguous()

        y = torch.cat((y1, y2, y3, y4), dim=-1)

        y = self.out_norm(y).view(b, q, q, -1)
        y = y * z
        out = self.out_proj(y)

        return out

class Multihead_Mamba(nn.Module):
    def __init__(self):
        super(Multihead_Mamba, self).__init__()
        self.mamba = MAMBA(256)
        self.norm = nn.LayerNorm(256)
        self.mlp = nn.Sequential(
            nn.Linear(256,256*4),
            nn.GELU(),
            nn.Linear(256*4,256)
        )

    def forward(self, x):
        result1 = x + self.mamba(self.norm(x))
        result = result1 + self.mlp(self.norm(result1))

        return result

class Get_G(nn.Module):
    def __init__(self):
        super(Get_G, self).__init__()
        self.mlp = nn.Linear(16*2, 16)
        self.sigmoid = nn.Sigmoid()

    def forward(self, A: torch.tensor, B: torch.tensor):  # B,L,D
        combined = torch.cat((A, B), 2)

        G = self.sigmoid(self.mlp(combined))  # (B,C) 每个通道一个系数

        return G

class Fusion(nn.Module):
    def __init__(self):
        super(Fusion, self).__init__()
        self.get_G = Get_G()

    def forward(self, A_in, B):  # B,L,D  B,H*W,C

        G = self.get_G(A_in, B)  # G:B,L,D
        A = A_in * G

        outer_product = torch.einsum('bci,bcj->bcij', A, B)  # (b, h*w, c, c)

        # Step 2: 对外积矩阵进行池化
        pooled_outer_product = torch.mean(outer_product, dim=2)  # (b,h*w,c,c) -> (b,h*w,c)每个外积矩阵平均池化

        # out = pooled_outer_product.permute(0, 2, 1)

        # return out
        return pooled_outer_product

class Block2_2(nn.Module):  # in:B,L,D  out:B,L,D
    def __init__(self):
        super(Block2_2, self).__init__()
        self.fusion_2 = Fusion()
        self.norm = nn.LayerNorm(16)

    def forward(self, mss, pans):  # B,16,16,64
        result = self.fusion_2(self.norm(mss), self.norm(pans))

        return result

class Net(nn.Module):  # 总参数700w  提取部分630w M (其中Downsample 50w M  Block1 580w M)  融合+分类 70w M
    def __init__(self,channel_ms,channel_pan, Classes):
        super(Net, self).__init__()

        self.upms = upms()
        self.R1 = R1()
        self.scan_161664_ms = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=256, kernel_size=8, stride=8, bias=True), Permute(0, 2, 3, 1),
            nn.LayerNorm(256))
        self.scan_161664_pan = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=256, kernel_size=8, stride=8, bias=True), Permute(0, 2, 3, 1),
            nn.LayerNorm(256))
        self.mamba_ms_1 = Multihead_Mamba()  # in:B,16,16,64  out:B,16,16,64
        self.mamba_pan_1 = Multihead_Mamba()
        self.FusionBlock2_2 = Block2_2()
        self.classifier = Classifier(Classes=Classes)

    def forward(self, ms_in, pan_in):
        ms = self.upms(ms_in)  # ms:64*4*16*16 -> 64*4*64*64
        pan = self.R1(pan_in)  # pan:64*1*64*64 -> 64*4*64*64
        b, c, h, w = ms.shape
        ms_patch = self.scan_161664_ms(ms)  # B,4,64,64 -> B,16,16,64
        pan_patch = self.scan_161664_pan(pan)  # B,4,64,64 -> B,16,16,64

        msf1 = self.mamba_ms_1(ms_patch)
        panf1 = self.mamba_pan_1(pan_patch)

        msf1 = msf1.reshape(b,16,16,16,2,2).permute(0,1,4,2,5,3).reshape(b, 32,32, 16).reshape(b,32*32,16)
        panf1 = panf1.reshape(b,16,16,16,2,2).permute(0,1,4,2,5,3).reshape(b, 32,32, 16).reshape(b,32*32,16)
        fu3 = self.FusionBlock2_2(msf1,panf1)

        fu4 = fu3.contiguous().view(b, 16 * 16 * 64)
        output = self.classifier(fu4)

        return output, ms_patch, pan_patch


if __name__ == "__main__":
    device = torch.device("cuda:0")
    net = Net(4,1,Classes=7)
    net = net.to(device)
    x1 = torch.randn(16, 4, 16, 16, device=device)
    x2 = torch.randn(16, 1, 64, 64, device=device)

    output  = net(x1, x2)
    print(output.shape)
    print(type(output))
    print(f"Number of parameters: {sum(p.numel() for p in net.parameters()):.2f}")
