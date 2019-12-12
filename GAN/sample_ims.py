import torch
from vis import render_3d_animation
import torch.nn as nn
import numpy as np

chkpt_path = './checkpoint/bonebothconvdiscrim81_60000.pt'
device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
latent_dim = 10
pose_dim = 51
batch_size = 5
rf = 81

def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.dense1 = nn.Linear(pose_dim, 128)
        self.dense2 = nn.Linear(128, 100)
        self.dense3 = nn.LSTM(100, 100, batch_first=True)#nn.Linear(1000, 1000)
        self.dense4 = nn.LSTM(100, 100, batch_first=True)
        self.dense5 = nn.LSTM(100, 100, batch_first=True)
        self.dense6 = nn.Linear(100, pose_dim)
    def forward(self, real_single, to_print):
        x = tile(real_single.unsqueeze(1), 1, rf)
        y = self.dense1(x)
        y = nn.functional.leaky_relu(y)
        y = self.dense2(y)
        y = nn.functional.leaky_relu(y)
        y, h = self.dense3(y)
        #x = nn.functional.leaky_relu(x)
        y, h = self.dense4(y)
        y, h = self.dense5(y)
        y = self.dense6(y)
        y = torch.reshape(y, (batch_size, rf, pose_dim))
        if to_print:
            print("Y IS: " + str(y))
        x = x + y
        return x

gen = Generator().to(device)
gen.load_state_dict(torch.load(chkpt_path))
gen.eval()
from load_data import DataGenerator

data_generator = DataGenerator(latent_dim)
latent_sample, real_sample = data_generator.sample_depth_batch('discriminator', batch_size, rf)
#latent_sample = torch.from_numpy(latent_sample).float().to(device)
real_sample = torch.from_numpy(real_sample[:,10,:]).float().to(device)
fake = gen(real_sample, False).detach().cpu().numpy()
real_sample = real_sample.unsqueeze(1).detach().cpu().numpy()
for i in range(batch_size):
    print("MAKING VIDEO " + str(i))
    render_3d_animation(np.expand_dims(real_sample[i], axis=0), ('boneconvdepthreal_{}.mp4').format(i), depth=True, limit=1)
    render_3d_animation(np.expand_dims(fake[i], axis=0), ('boneconvdepthsampled_{}.mp4').format(i), depth=True, limit=rf)
    render_3d_animation(np.expand_dims(fake[i], axis=0), ('boneconvdepthnorm_{}.mp4').format(i), depth=True, limit=rf, norm=False)
