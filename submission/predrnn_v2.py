
from argparse import Namespace
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
# class SpatioTemporalLSTMCell(nn.Module):
#     def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
#         super(SpatioTemporalLSTMCell, self).__init__()

#         self.num_hidden = num_hidden
#         self.padding = filter_size // 2
#         self._forget_bias = 1.0
#         if layer_norm:
#             self.conv_x = nn.Sequential(
#                 nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
#                 nn.LayerNorm([num_hidden * 7, width, width])
#             )
#             self.conv_h = nn.Sequential(
#                 nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
#                 nn.LayerNorm([num_hidden * 4, width, width])
#             )
#             self.conv_m = nn.Sequential(
#                 nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
#                 nn.LayerNorm([num_hidden * 3, width, width])
#             )
#             self.conv_o = nn.Sequential(
#                 nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
#                 nn.LayerNorm([num_hidden, width, width])
#             )
#         else:
#             self.conv_x = nn.Sequential(
#                 nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
#             )
#             self.conv_h = nn.Sequential(
#                 nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
#             )
#             self.conv_m = nn.Sequential(
#                 nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
#             )
#             self.conv_o = nn.Sequential(
#                 nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
#             )
#         self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)


#     def forward(self, x_t, h_t, c_t, m_t):
#         x_concat = self.conv_x(x_t)
#         h_concat = self.conv_h(h_t)
#         m_concat = self.conv_m(m_t)
#         i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
#         i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
#         i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

#         i_t = torch.sigmoid(i_x + i_h)
#         f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
#         g_t = torch.tanh(g_x + g_h)

#         delta_c = i_t * g_t
#         c_new = f_t * c_t + delta_c

#         i_t_prime = torch.sigmoid(i_x_prime + i_m)
#         f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
#         g_t_prime = torch.tanh(g_x_prime + g_m)

#         delta_m = i_t_prime * g_t_prime
#         m_new = f_t_prime * m_t + delta_m

#         mem = torch.cat((c_new, m_new), 1)
#         o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
#         h_new = o_t * torch.tanh(self.conv_last(mem))

#         return h_new, c_new, m_new, delta_c, delta_m

# class RNN(nn.Module):
#     def __init__(self, num_layers, num_hidden, configs):
#         super(RNN, self).__init__()

#         self.configs = configs
#         self.visual = self.configs.visual
#         self.visual_path = self.configs.visual_path

#         self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
#         self.num_layers = num_layers
#         self.num_hidden = num_hidden
#         cell_list = []

#         width = configs.img_width // configs.patch_size
#         self.MSE_criterion = nn.MSELoss()

#         for i in range(num_layers):
#             in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
#             cell_list.append(
#                 SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
#                                        configs.stride, configs.layer_norm)
#             )
#         self.cell_list = nn.ModuleList(cell_list)
#         self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel, kernel_size=1, stride=1, padding=0,
#                                    bias=False)
#         # shared adapter
#         adapter_num_hidden = num_hidden[0]
#         self.adapter = nn.Conv2d(adapter_num_hidden, adapter_num_hidden, 1, stride=1, padding=0, bias=False)

#     def forward(self, frames_tensor, mask_true):
#         # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
#         frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
#         mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

#         # [batch, length, channel, height, width]
#         # frames = frames_tensor.contiguous()
#         batch = frames.shape[0]
#         height = frames.shape[3]
#         width = frames.shape[4]

#         next_frames = []
#         h_t = []
#         c_t = []
#         delta_c_list = []
#         delta_m_list = []
#         if self.visual:
#             delta_c_visual = []
#             delta_m_visual = []

#         decouple_loss = []

#         for i in range(self.num_layers):
#             zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
#             h_t.append(zeros)
#             c_t.append(zeros)
#             delta_c_list.append(zeros)
#             delta_m_list.append(zeros)

#         memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

#         for t in range(self.configs.total_length - 1):
#             if self.configs.reverse_scheduled_sampling == 1:
#                 # reverse schedule sampling
#                 if t == 0:
#                     net = frames[:, t]
#                 else:
#                     net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
#             else:
#                 # schedule sampling
#                 if t < self.configs.input_length:
#                     net = frames[:, t]
#                 else:
#                     net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
#                           (1 - mask_true[:, t - self.configs.input_length]) * x_gen

#             h_t[0], c_t[0], memory, delta_c, delta_m = self.cell_list[0](net, h_t[0], c_t[0], memory)
#             delta_c_list[0] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
#             delta_m_list[0] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
#             if self.visual:
#                 delta_c_visual.append(delta_c.view(delta_c.shape[0], delta_c.shape[1], -1))
#                 delta_m_visual.append(delta_m.view(delta_m.shape[0], delta_m.shape[1], -1))

#             for i in range(1, self.num_layers):
#                 h_t[i], c_t[i], memory, delta_c, delta_m = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)
#                 delta_c_list[i] = F.normalize(self.adapter(delta_c).view(delta_c.shape[0], delta_c.shape[1], -1), dim=2)
#                 delta_m_list[i] = F.normalize(self.adapter(delta_m).view(delta_m.shape[0], delta_m.shape[1], -1), dim=2)
#                 if self.visual:
#                     delta_c_visual.append(delta_c.view(delta_c.shape[0], delta_c.shape[1], -1))
#                     delta_m_visual.append(delta_m.view(delta_m.shape[0], delta_m.shape[1], -1))

#             x_gen = self.conv_last(h_t[self.num_layers - 1])
#             next_frames.append(x_gen)
#             # decoupling loss
#             for i in range(0, self.num_layers):
#                 decouple_loss.append(
#                     torch.mean(torch.abs(torch.cosine_similarity(delta_c_list[i], delta_m_list[i], dim=2))))


#         decouple_loss = torch.mean(torch.stack(decouple_loss, dim=0))
#         # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
#         next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
#         # loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:]) + self.configs.decouple_beta * decouple_loss #change to msssim loss and move loss out of this func (might need to fiddle with hyperparams) 
#         loss = self.configs.decouple_beta * decouple_loss #change to msssim loss and move loss out of this func (might need to fiddle with hyperparams) 
#         return next_frames, loss



class SpatioTemporalLSTMCell(nn.Module):
    def __init__(self, in_channel, num_hidden, width, filter_size, stride, layer_norm):
        super(SpatioTemporalLSTMCell, self).__init__()

        self.num_hidden = num_hidden
        self.padding = filter_size // 2
        self._forget_bias = 1.0
        if layer_norm:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 7, width, width])
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 4, width, width])
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden * 3, width, width])
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
                nn.LayerNorm([num_hidden, width, width])
            )
        else:
            self.conv_x = nn.Sequential(
                nn.Conv2d(in_channel, num_hidden * 7, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_h = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 4, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_m = nn.Sequential(
                nn.Conv2d(num_hidden, num_hidden * 3, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
            self.conv_o = nn.Sequential(
                nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=filter_size, stride=stride, padding=self.padding, bias=False),
            )
        self.conv_last = nn.Conv2d(num_hidden * 2, num_hidden, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x_t, h_t, c_t, m_t):
        x_concat = self.conv_x(x_t)
        h_concat = self.conv_h(h_t)
        m_concat = self.conv_m(m_t)
        i_x, f_x, g_x, i_x_prime, f_x_prime, g_x_prime, o_x = torch.split(x_concat, self.num_hidden, dim=1)
        i_h, f_h, g_h, o_h = torch.split(h_concat, self.num_hidden, dim=1)
        i_m, f_m, g_m = torch.split(m_concat, self.num_hidden, dim=1)

        i_t = torch.sigmoid(i_x + i_h)
        f_t = torch.sigmoid(f_x + f_h + self._forget_bias)
        g_t = torch.tanh(g_x + g_h)

        c_new = f_t * c_t + i_t * g_t

        i_t_prime = torch.sigmoid(i_x_prime + i_m)
        f_t_prime = torch.sigmoid(f_x_prime + f_m + self._forget_bias)
        g_t_prime = torch.tanh(g_x_prime + g_m)

        m_new = f_t_prime * m_t + i_t_prime * g_t_prime

        mem = torch.cat((c_new, m_new), 1)
        o_t = torch.sigmoid(o_x + o_h + self.conv_o(mem))
        h_new = o_t * torch.tanh(self.conv_last(mem))

        return h_new, c_new, m_new



class RNN(nn.Module):
    def __init__(self, num_layers, num_hidden, configs):
        super(RNN, self).__init__()

        self.configs = configs
        self.frame_channel = configs.patch_size * configs.patch_size * configs.img_channel
        self.num_layers = num_layers
        self.num_hidden = num_hidden
        cell_list = []

        width = configs.img_width // configs.patch_size
        # self.MSE_criterion = nn.MSELoss()

        for i in range(num_layers):
            in_channel = self.frame_channel if i == 0 else num_hidden[i - 1]
            cell_list.append(
                SpatioTemporalLSTMCell(in_channel, num_hidden[i], width, configs.filter_size,
                                       configs.stride, configs.layer_norm)
            )
        self.cell_list = nn.ModuleList(cell_list)
        self.conv_last = nn.Conv2d(num_hidden[num_layers - 1], self.frame_channel,
                                   kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, frames_tensor, mask_true):
        # [batch, length, height, width, channel] -> [batch, length, channel, height, width]
        frames = frames_tensor.permute(0, 1, 4, 2, 3).contiguous()
        mask_true = mask_true.permute(0, 1, 4, 2, 3).contiguous()

        batch = frames.shape[0]
        height = frames.shape[3]
        width = frames.shape[4]

        next_frames = []
        h_t = []
        c_t = []

        for i in range(self.num_layers):
            zeros = torch.zeros([batch, self.num_hidden[i], height, width]).to(self.configs.device)
            h_t.append(zeros)
            c_t.append(zeros)

        memory = torch.zeros([batch, self.num_hidden[0], height, width]).to(self.configs.device)

        for t in range(self.configs.total_length - 1):
            # reverse schedule sampling
            if self.configs.reverse_scheduled_sampling == 1:
                if t == 0:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - 1] * frames[:, t] + (1 - mask_true[:, t - 1]) * x_gen
            else:
                if t < self.configs.input_length:
                    net = frames[:, t]
                else:
                    net = mask_true[:, t - self.configs.input_length] * frames[:, t] + \
                          (1 - mask_true[:, t - self.configs.input_length]) * x_gen

            h_t[0], c_t[0], memory = self.cell_list[0](net, h_t[0], c_t[0], memory)

            for i in range(1, self.num_layers):
                h_t[i], c_t[i], memory = self.cell_list[i](h_t[i - 1], h_t[i], c_t[i], memory)

            x_gen = self.conv_last(h_t[self.num_layers - 1])
            next_frames.append(x_gen)

        # [length, batch, channel, height, width] -> [batch, length, height, width, channel]
        next_frames = torch.stack(next_frames, dim=0).permute(1, 0, 3, 4, 2).contiguous()
        # loss = self.MSE_criterion(next_frames, frames_tensor[:, 1:])
        return next_frames, 0

class PredRNNModel(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        self.iter = 0
        self.eta = config['sampling_start_value']
        config['input_length'] = config['inputs']
        config['total_length'] = config['inputs'] + config['outputs']
        self.args = Namespace(**config)
        # self.args.device = torch.device('cuda')
        self.predrnn = RNN(len(config['num_hidden']), config['num_hidden'], self.args)
        self.config = config

    def forward(self, inp, override=False):
        x = rearrange(inp, 'b (c t) h w -> b t h w c', c=1)
        x = reshape_patch(x, self.config['patch_size']).float()
        old_eta = self.eta
        self.eta, real_mask = schedule_sampling(self.eta, self.iter, self.args, scheduled_sampling=(self.training or override)) 
        if not self.training:
            self.eta = old_eta
        real_mask = torch.tensor(real_mask).float().to(self.args.device)
        
        x, loss = self.predrnn(x, real_mask)
        x = reshape_patch_back(x, self.config['patch_size']).float()
        x = rearrange(x, 'b t h w c -> b (c t) h w')
        if self.training:
            self.iter += 1
        # return x[:, 1:], loss
        return x, loss

    # def update(self):
        # self.iter += 1


def reshape_patch_back(patch_tensor, patch_size):
    assert 5 == patch_tensor.ndim
    a = rearrange(patch_tensor, 'b t h w (c p1 p2) -> b t (h p1) (w p2) c', p1=patch_size, p2=patch_size)
    return a

def reshape_patch(img_tensor, patch_size):
    assert 5 == img_tensor.ndim
    a = rearrange(img_tensor, 'b t (h p1) (w p2) c -> b t h w (c p1 p2)', p1=patch_size, p2=patch_size)
    return a
import numpy as np
def schedule_sampling(eta, itr, args, scheduled_sampling=True):
    zeros = np.zeros((args.batch_size,
                      args.total_length - args.input_length - 1,
                      args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    if (not args.scheduled_sampling) and (not scheduled_sampling):
        return 0.0, zeros

    if itr < args.sampling_stop_iter:
        eta -= args.sampling_changing_rate
    else:
        eta = 0.0
    random_flip = np.random.random_sample(
        (args.batch_size, args.total_length - args.input_length - 1))
    true_token = (random_flip < eta)
    ones = np.ones((args.img_width // args.patch_size,
                    args.img_width // args.patch_size,
                    args.patch_size ** 2 * args.img_channel))
    zeros = np.zeros((args.img_width // args.patch_size,
                      args.img_width // args.patch_size,
                      args.patch_size ** 2 * args.img_channel))
    real_input_flag = []
    for i in range(args.batch_size):
        for j in range(args.total_length - args.input_length - 1):
            if true_token[i, j]:
                real_input_flag.append(ones)
            else:
                real_input_flag.append(zeros)
    real_input_flag = np.array(real_input_flag)
    real_input_flag = np.reshape(real_input_flag,
                                 (args.batch_size,
                                  args.total_length - args.input_length - 1,
                                  args.img_width // args.patch_size,
                                  args.img_width // args.patch_size,
                                  args.patch_size ** 2 * args.img_channel))
    return eta, real_input_flag