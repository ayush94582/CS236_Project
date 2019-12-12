import numpy as np
import random
from load_data import DataGenerator
from tensorflow.python.platform import flags
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vis import render_3d_animation

initial_K_D = 0
initial_K_G = 0
combined = 1000000
sampling = 'supervised27given'

batch_size = 64
latent_dim = 10
pose_dim = 51
rf = 81
hidden_size = 128
frames_given = 27

FLAGS = flags.FLAGS

flags.DEFINE_integer(
    'generator_RF', 81, 'number of classes used in classification (e.g. 5-way classification).')

flags.DEFINE_integer('discriminator_RF', 81,
                     'number of examples used for inner gradient update (K for K-shot learning).')

device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

def tile(a, dim, n_tile):
        init_dim = a.size(dim)
        repeat_idx = [1] * a.dim()
        repeat_idx[dim] = n_tile
        a = a.repeat(*(repeat_idx))
        order_index = torch.LongTensor(np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])).to(device)
        return torch.index_select(a, dim, order_index)

def bone_loss(fake):
# fake shape is (batch_size, rf, 51)
    fake = torch.reshape(fake, (batch_size, rf, 17, 3))
    joints1 = [1,2,4,5,11,12,14,15]
    joints2 = [2,3,5,6,12,13,15,16]
    dists = torch.norm(fake[:,:,joints1,:] - fake[:,:,joints2,:], dim=3) # shape of (batch_size, rf, 8)
    averaged = tile(torch.mean(dists, dim=1).unsqueeze(1), 1, rf)
    diff = torch.mean(torch.abs(dists - averaged)) * len(joints1) * 10
    return diff

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
        y = self.dense1(real_single)
        y = nn.functional.leaky_relu(y)
        y = self.dense2(y)
        y = nn.functional.leaky_relu(y)
        y, h = self.dense3(y)
        #x = nn.functional.leaky_relu(x)
        y, h = self.dense4(y)
        y, h = self.dense5(y)
        y = self.dense6(y)
        y = torch.reshape(y, (batch_size, rf, 17, 3))
        y = torch.cumsum(y, axis=2)
        y = torch.reshape(y, (batch_size, rf, pose_dim))
        if to_print:
            print("Y IS: " + str(y))
        x = torch.cat((real_single[:, 0:frames_given], real_single[:, frames_given:] + y[:, frames_given:]), dim=1)
        #x = real_single + y
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv1d(pose_dim, hidden_size, 3, dilation=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, dilation=3)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, 1, dilation=1)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, 3, dilation=9)
        self.conv5 = nn.Conv1d(hidden_size, hidden_size, 1, dilation=1)
        self.conv6 = nn.Conv1d(hidden_size, hidden_size, 3, dilation=27)
        self.conv7 = nn.Conv1d(hidden_size, hidden_size, 1, dilation=1)
        self.conv8 = nn.Conv1d(hidden_size, 1, 1, dilation=1)
        self.act_fn = nn.Sigmoid()
    def forward(self, x):
        x = torch.reshape(x, (batch_size, rf, pose_dim)).permute(0,2,1)
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv3(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv4(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv5(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv6(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv7(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv8(x)
        x = self.act_fn(x)
        return x

class Discriminator2D(nn.Module):
    def __init__(self):
        super(Discriminator2D, self).__init__()
        self.conv1 = nn.Conv1d(34, hidden_size, 3, dilation=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, 3, dilation=3)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, 1, dilation=1)
        self.conv4 = nn.Conv1d(hidden_size, hidden_size, 3, dilation=9)
        self.conv5 = nn.Conv1d(hidden_size, hidden_size, 1, dilation=1)
        self.conv6 = nn.Conv1d(hidden_size, hidden_size, 3, dilation=27)
        self.conv7 = nn.Conv1d(hidden_size, hidden_size, 1, dilation=1)
        self.conv8 = nn.Conv1d(hidden_size, 1, 1, dilation=1)
        self.act_fn = nn.Sigmoid()
    def forward(self, x):
        x = torch.reshape(x, (batch_size, rf, 34)).permute(0,2,1)
        x = self.conv1(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv2(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv3(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv4(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv5(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv6(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv7(x)
        x = nn.functional.leaky_relu(x)
        x = self.conv8(x)
        x = self.act_fn(x)
        return x

data_generator = DataGenerator(latent_dim)
criterion = nn.BCELoss()
real_label = 1
fake_label = 0

netG = Generator().to(device)
print(netG)
netD = Discriminator().to(device)
print(netD)
net2D = Discriminator2D().to(device)
print(net2D)

optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizer2D = optim.Adam(net2D.parameters(), lr=0.0002, betas=(0.5,0.999))

def mpjpe(predicted, target):
    """
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=len(target.shape)-1))

def getAcc(real_preds, fake_preds):
    print("REAL PREDS ARE: " + str(real_preds))
    print("FAKE PREDS ARE: " + str(fake_preds))
    pos = torch.ones(batch_size).to(device)
    neg = torch.zeros(batch_size).to(device)
    real = torch.where(real_preds > 0.5, pos, neg)
    fake = torch.where(fake_preds < 0.5, pos, neg)
    return (torch.sum(real) + torch.sum(fake) + 0.0)/(2 * batch_size + 0.0)

gen_loss = []
dis_loss = []
print('Started adversarial training for {} steps'.format(combined))
for adv_step in range(combined):
    if adv_step % 20000 == 0:
        torch.save(netG.state_dict(), ('./checkpoint/' + str(sampling) + '_{}.pt').format(adv_step))
        torch.save(netD.state_dict(), ('./checkpoint/' + str(sampling) + '_d_{}.pt').format(adv_step))
    latent_sample, real_sample = data_generator.sample_batch('discrim', batch_size, FLAGS.discriminator_RF)
    #latent_sample, real_sample, real_2d_sample = data_generator.sample_both_depth_batch('discrim', batch_size, FLAGS.discriminator_RF)
    sing_real = np.concatenate((real_sample[:,0:frames_given,:], np.repeat(np.expand_dims(real_sample[:,frames_given-1,:], axis=1), rf - frames_given, 1)), axis=1)
    sing_real = torch.from_numpy(sing_real).float().to(device)
    latent_sample = torch.from_numpy(latent_sample).float().to(device)
    real_sample = torch.from_numpy(real_sample).float().to(device)
    #real_2d_sample = torch.from_numpy(real_2d_sample).float().to(device)
    to_print = False
    for i in range(1):
        netG.zero_grad()
        label = torch.full((batch_size,), real_label, device=device)
        fake = netG(sing_real, to_print)
        #fake_2d = torch.reshape(torch.reshape(fake, (batch_size, rf, 17, 3))[:,:,:,:2], (batch_size, rf, 34))
        d_output = netD(fake).view(-1)
        #twod_output = net2D(fake_2d).view(-1)
        errG1 = criterion(d_output, label)
        #errG2 = criterion(twod_output, label)
        errG2 = 3*mpjpe(torch.reshape(real_sample[:,frames_given:,:], (batch_size, rf-frames_given, 17, 3)), torch.reshape(fake[:,frames_given:,:], (batch_size, rf-frames_given, 17, 3)))
        errG3 = bone_loss(fake)
        errG = errG1 + errG2 + errG3
        errG.backward()
        optimizerG.step()
        if adv_step % 1000 == 0:
            render_3d_animation(fake.cpu().detach().numpy(), (str(sampling)+'_gen_{}.mp4').format(adv_step), limit=rf, depth=False, traj=False)
            print("saved video at combined epoch {}".format(adv_step))
        if adv_step % 250 == 0:
            #print("FAKE IS: " + str(fake))
            gen_loss.append(errG)
            print('Generator Epoch {} : Loss {}, Loss1 {}, Loss2 {}, Bone {}'.format(adv_step, errG, errG1, errG2, errG3))
    for i in range(1):
        netD.zero_grad()
        label = torch.full((batch_size,), real_label, device=device)
        pred_D = netD(real_sample).view(-1)
        errD_real = criterion(pred_D, label)
        errD_real.backward()
        fake = netG(sing_real, False)
        pred_fake = netD(fake.detach()).view(-1)
        label.fill_(fake_label)
        errD_fake = criterion(pred_fake, label)
        errD_fake.backward()
        errD = errD_real + errD_fake
        optimizerD.step()
        if adv_step % 250 == 0:
            acc = getAcc(pred_D, pred_fake)
            dis_loss.append(errD)
            print('Discrim Epoch {} : Loss {} : Accuracy {}'.format(adv_step, errD, acc))
    #for i in range(1):
    #    net2D.zero_grad()
    #    label = torch.full((batch_size,), real_label, device=device)
    #    pred_2D = net2D(real_2d_sample).view(-1)
    #    err2D_real = criterion(pred_2D, label)
    #    err2D_real.backward()
    #    fake = torch.reshape(torch.reshape(netG(sing_real, False), (batch_size, rf, 17, 3))[:,:,:,:2], (batch_size, rf, 34))
    #    pred_fake = net2D(fake.detach()).view(-1)
    #    label.fill_(fake_label)
    #    err2D_fake = criterion(pred_fake, label)
    #    err2D_fake.backward()
    #    err2D = err2D_real + err2D_fake
    #    optimizer2D.step()
    #    if adv_step % 250 == 0:
    #        acc = getAcc(pred_2D, pred_fake)
    #        print('Two-Discrim Epoch {} : Loss {} : Accuracy {}'.format(adv_step, err2D, acc))

#np.savez('./traj_losses.npz', gen=np.asarray(gen_loss), dis=np.asarray(dis_loss))
