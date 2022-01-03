import math
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.distributions as D
import torch.nn.functional as F

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
font = {'family' : 'sans-serif',
        'weight' : 'bold',
        'size'   : 16}
matplotlib.rc('font', **font)

# ------------------------------
# FCE Value Function 
# ------------------------------
def value(energy, flow, x, z, maximize=True):
    gen, logq_gen = flow.inverse(z)  # the second term is logq(x̃)
    logp_x = energy(x)  # logp(x)
    logp_gen = energy(gen)  #logp(x̃)
    logq_x = flow.log_prob(x)  # logq(x)

    value_x = logp_x - torch.logsumexp(torch.cat([logp_x, logq_x], dim=1), dim=1, keepdim=True)  # logp(x)/(logp(x) + logq(x))
    value_gen = logq_gen - torch.logsumexp(torch.cat([logp_gen, logq_gen], dim=1), dim=1, keepdim=True) # logq(x̃)/(logp(x̃) + logq(x̃))
    
    v = value_x.mean() + value_gen.mean()

    # calculate accuracy
    r_x = torch.sigmoid(logp_x - logq_x)
    r_gen = torch.sigmoid(logq_gen - logp_gen)
    acc = ((r_x >= 1/2).sum() + (r_gen > 1/2).sum()).cpu().numpy() / (len(r_x) + len(r_gen))

    if maximize:
        return -v,  acc
    else:
        return v, acc

# ------------------------------
# MIXED GAUSSIAN DENSITY 
# ------------------------------
class MixedGaussian():
    def __init__(self, device):
        r = 2. * math.sqrt(2)
        self.covariance_matrix = (1/8 * torch.eye(2)).to(device)  # variance on the diagonal, not standard deviation
        self.m1 = D.MultivariateNormal(loc=torch.tensor([r, 0.], device=device), covariance_matrix=self.covariance_matrix)
        self.m2 = D.MultivariateNormal(loc=torch.tensor([-r, 0.], device=device), covariance_matrix=self.covariance_matrix)
        self.m3 = D.MultivariateNormal(loc=torch.tensor([0., r], device=device), covariance_matrix=self.covariance_matrix)
        self.m4 = D.MultivariateNormal(loc=torch.tensor([0., -r], device=device), covariance_matrix=self.covariance_matrix)
        self.m5 = D.MultivariateNormal(loc=torch.tensor([2., 2.], device=device), covariance_matrix=self.covariance_matrix)
        self.m6 = D.MultivariateNormal(loc=torch.tensor([-2., 2.], device=device), covariance_matrix=self.covariance_matrix)
        self.m7 = D.MultivariateNormal(loc=torch.tensor([2., -2.], device=device), covariance_matrix=self.covariance_matrix)
        self.m8 = D.MultivariateNormal(loc=torch.tensor([-2., -2.], device=device), covariance_matrix=self.covariance_matrix)

    def log_prob(self, x):
        log_probs = torch.cat([self.m1.log_prob(x).unsqueeze(1), 
                               self.m2.log_prob(x).unsqueeze(1), 
                               self.m3.log_prob(x).unsqueeze(1), 
                               self.m4.log_prob(x).unsqueeze(1),
                               self.m5.log_prob(x).unsqueeze(1), 
                               self.m6.log_prob(x).unsqueeze(1), 
                               self.m7.log_prob(x).unsqueeze(1), 
                               self.m8.log_prob(x).unsqueeze(1)], dim=1)
        out = - math.log(8.) + torch.logsumexp(log_probs, dim=1, keepdim=True)
        return out

# MSE between EBM and a true distribution.
def mse(ebm, true_dist, data_batch):
    with torch.no_grad():
        ebm_outputs = ebm(data_batch)
        true_values = true_dist.log_prob(data_batch)
        return F.mse_loss(input=ebm_outputs, target=true_values).item()

#-------------------------------------------
# DATA
#-------------------------------------------
def get_data(args):
    dataset = sample_2d_data(dataset=args.dataset, n_samples=args.samples)
    dataloader  = DataLoader(dataset, batch_size=args.batch, shuffle=True)
    return dataset, dataloader
    

def sample_2d_data(dataset='8gaussians', n_samples=50000):
    
    z = torch.randn(n_samples, 2)

    if dataset == '8gaussians':
        scale = 4
        sq2 = 1/math.sqrt(2)
        centers = [(1,0), (-1,0), (0,1), (0,-1), (sq2,sq2), (-sq2,sq2), (sq2,-sq2), (-sq2,-sq2)]
        centers = torch.tensor([(scale * x, scale * y) for x,y in centers])
        return sq2 * (0.5 * z + centers[torch.randint(len(centers), size=(n_samples,))])
        # return  0.05 * z + centers[torch.randint(len(centers), size=(n_samples,))]

    elif dataset == '2spirals':
        n = torch.sqrt(torch.rand(n_samples // 2)) * 540 * (2 * math.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples // 2) * 0.5
        d1y =   torch.sin(n) * n + torch.rand(n_samples // 2) * 0.5
        x = torch.cat([torch.stack([ d1x,  d1y], dim=1),
                       torch.stack([-d1x, -d1y], dim=1)], dim=0) / 3
        return x + 0.1*z

    elif dataset == 'spiral':
        n = torch.sqrt(torch.rand(n_samples)) * 540 * (2 * math.pi) / 360
        d1x = - torch.cos(n) * n + torch.rand(n_samples) * 0.5
        d1y =   torch.sin(n) * n + torch.rand(n_samples) * 0.5
        print(d1x.shape)
        x = torch.stack([ d1x,  d1y], dim=1) / 3
        print(x.shape)
        return x + 0.1*z

    elif dataset == 'checkerboard':
        x1 = torch.rand(n_samples) * 4 - 2
        x2_ = torch.rand(n_samples) - torch.randint(0, 2, (n_samples,), dtype=torch.float) * 2
        x2 = x2_ + x1.floor() % 2
        return torch.stack([x1, x2], dim=1) * 2

    elif dataset == 'rings':
        n_samples4 = n_samples3 = n_samples2 = n_samples // 4
        n_samples1 = n_samples - n_samples4 - n_samples3 - n_samples2
        # so as not to have the first point = last point, set endpoint=False in np; here shifted by one
        linspace4 = torch.linspace(0, 2 * math.pi, n_samples4 + 1)[:-1]
        linspace3 = torch.linspace(0, 2 * math.pi, n_samples3 + 1)[:-1]
        linspace2 = torch.linspace(0, 2 * math.pi, n_samples2 + 1)[:-1]
        linspace1 = torch.linspace(0, 2 * math.pi, n_samples1 + 1)[:-1]
        circ4_x = torch.cos(linspace4)
        circ4_y = torch.sin(linspace4)
        circ3_x = torch.cos(linspace4) * 0.75
        circ3_y = torch.sin(linspace3) * 0.75
        circ2_x = torch.cos(linspace2) * 0.5
        circ2_y = torch.sin(linspace2) * 0.5
        circ1_x = torch.cos(linspace1) * 0.25
        circ1_y = torch.sin(linspace1) * 0.25

        x = torch.stack([torch.cat([circ4_x, circ3_x, circ2_x, circ1_x]),
                         torch.cat([circ4_y, circ3_y, circ2_y, circ1_y])], dim=1) * 3.0

        # random sample
        x = x[torch.randint(0, n_samples, size=(n_samples,))]

        # Add noise
        return x + torch.normal(mean=torch.zeros_like(x), std=0.08*torch.ones_like(x))

    elif dataset == "pinwheel":
        rng = np.random.RandomState()
        radial_std = 0.3
        tangential_std = 0.1
        num_classes = 5
        num_per_class = n_samples // 5
        rate = 0.25
        rads = np.linspace(0, 2 * np.pi, num_classes, endpoint=False)
        features = rng.randn(num_classes*num_per_class, 2) * np.array([radial_std, tangential_std])
        # features = np.random.randn(num_classes*num_per_class, 2) * np.array([radial_std, tangential_std])
        features[:, 0] += 1.
        labels = np.repeat(np.arange(num_classes), num_per_class)

        angles = rads[labels] + rate * np.exp(features[:, 0])
        rotations = np.stack([np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles)])
        rotations = np.reshape(rotations.T, (-1, 2, 2))
        
        data = 2 * rng.permutation(np.einsum("ti,tij->tj", features, rotations))
        # data = 2 * np.random.permutation(np.einsum("ti,tij->tj", features, rotations))
        return torch.as_tensor(data, dtype=torch.float32)

    else:
        raise RuntimeError('Invalid `dataset` to sample from.')

# --------------------
# Plotting
# --------------------

@torch.no_grad()
def plot(dataset, energy, flow, epoch, device):
    n_pts = 1000
    range_lim = 4
    # construct test points
    test_grid = setup_grid(range_lim, n_pts, device)

    # plot
    fig, axs = plt.subplots(2, 2, figsize=(8,8), subplot_kw={'aspect': 'equal'})
    plot_samples(dataset, axs[0][0], range_lim, n_pts)
    plot_flow(flow, axs[0][1], test_grid, n_pts)
    plot_flow_samples(flow, axs[1][0], range_lim, n_pts)
    plot_energy(energy, axs[1][1], test_grid, n_pts)

    # format
    for ax in plt.gcf().axes: format_ax(ax, range_lim)
    plt.tight_layout(pad=2.0)

    # save
    print('Saving image to images/....')
    plt.savefig('images/epoch_{}.png'.format(epoch))
    plt.close()


def setup_grid(range_lim, n_pts, device):
    x = torch.linspace(-range_lim, range_lim, n_pts)
    xx, yy = torch.meshgrid((x, x))
    zz = torch.stack((xx.flatten(), yy.flatten()), dim=1)
    return xx, yy, zz.to(device)

def plot_samples(dataset, ax, range_lim, n_pts):
    samples = dataset.numpy()
    ax.hist2d(samples[:,0], samples[:,1], range=[[-range_lim, range_lim], [-range_lim, range_lim]], bins=n_pts, cmap=plt.cm.jet)
    ax.set_title('Data')

def plot_energy(energy, ax, test_grid, n_pts):
    xx, yy, zz = test_grid
    log_prob = energy(zz)
    prob = log_prob.exp().cpu()
    ax.pcolormesh(xx, yy, prob.view(n_pts,n_pts), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title('EBM')

def plot_flow(flow, ax, test_grid, n_pts):
    flow.eval()
    xx, yy, zz = test_grid
    log_prob = flow.log_prob(zz)
    prob = log_prob.exp().cpu()
    ax.pcolormesh(xx, yy, prob.view(n_pts,n_pts), cmap=plt.cm.jet)
    ax.set_facecolor(plt.cm.jet(0.))
    ax.set_title('Flow')

def plot_flow_samples(flow, ax, range_lim, n_pts):
    z = flow.base_dist.sample((10000,))
    samples, _ = flow.inverse(z)
    samples = samples.cpu().numpy()
    ax.hist2d(samples[:,0], samples[:,1], range=[[-range_lim, range_lim], [-range_lim, range_lim]], bins=n_pts, cmap=plt.cm.jet)
    ax.set_title('Flow-samples')

def format_ax(ax, range_lim):
    ax.set_xlim(-range_lim, range_lim)
    ax.set_ylim(-range_lim, range_lim)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.invert_yaxis()