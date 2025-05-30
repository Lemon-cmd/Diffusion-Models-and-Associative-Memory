"""2D toy example from the paper : Memorization to Generalization: Diffusion Models from Dense Associative Memory"""
import os
import math
import torch
import click
from tqdm import tqdm
import numpy as np
import torch.nn as nn
from itertools import cycle
import matplotlib.pyplot as plt
from scipy.special import i0, i1
from torch.utils.data import Dataset
from torch.utils.data import DataLoader 
import scipy.integrate as integrate
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial import distance_matrix
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
sns.set_theme(style="whitegrid")

#------------------------------------------------------------------------------------------------
# Data generation. It generates data that lies on a unit circle and allows sampling a subset of data.
# The class CircleData is a wrapper around the data generation function and has the exact energy of the model 

def generate_circle_data(num_samples=50000, radius=1, seed=59):
    np.random.seed(seed) 
    angles = np.random.uniform(0, 2*np.pi, num_samples)
    x = radius * np.cos(angles)
    y = radius * np.sin(angles)
    return np.stack([x, y], axis=1)


class CircleDataset(Dataset):
    def __init__(self, num_samples=50000, radius=1, seed=9):
        self.data = generate_circle_data(num_samples, radius, seed)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def cartesian_to_polar(self, samples):
        x = samples[:, 0]
        y = samples[:, 1]
        r = np.sqrt(x**2 + y**2)
        angles = np.arctan2(y, x)
        return r, angles
    
    def energy_am(self, samples, beta, normalize=True):
        """
        Computes the energy function E^AM(R, phi) = R^2 + 1 - (1 / beta) * log(I_0(2 * beta * R)). Eq. (13) in the paper.
        """
        r, _ = self.cartesian_to_polar(samples)
        energy = r**2 + 1 - (1 / beta) * np.log(i0(2 * beta * r))

        # Shift so that the lowest energy is 
        if normalize:
            min_energy = energy.min()
            energy -= min_energy 

        return energy

    def score_am(self, samples, beta, epsilon=1e-6):
        """
        Computes the score function S^AM(R, phi) = -2 * R - (2 / beta) * I_1(2 * beta * R) / I_0(2 * beta * R). Eq. (18) in the paper.
        """
        r, _ = self.cartesian_to_polar(samples)
        
        # Ensure r is not zero to avoid division by zero
        r = np.clip(r, epsilon, np.inf)
        
        bessel_ratio = i1(2 * beta * r) / i0(2 * beta * r)
  
        score_r = 2 * (bessel_ratio - r) 
        score = score_r[:, None] * samples / r[:, None]
        
        return score

#------------------------------------------------------------------------------------------------
# Data Utilities

def create_subset(dataset, sample_size, seed=42):
    """Create a subset of the dataset based on the specified sample size. """
    max_size = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    if not 1 <= sample_size <= len(dataset):
        raise ValueError("Sample size must be between 1 and the size of the dataset inclusive.")
    subset, _ = torch.utils.data.random_split(
        dataset, [sample_size, max_size - sample_size], generator=generator
    )
    return subset

def prepare_datasets(sample_size, train_size=60000, test_size=10000, seed=9):
    dataset = CircleDataset(num_samples=train_size, seed=seed)
    test_dataset = CircleDataset(num_samples=test_size, seed=seed)
    train_subset = create_subset(dataset, sample_size)
    test_subset = create_subset(test_dataset, sample_size)
    return train_subset, test_subset

#------------------------------------------------------------------------------------------------
# Visualization Helper Functions


def save_circle_plot(data, save_dir, filename='circle_plot.png', radius=1, figsize=(8, 8), fontsize=12, show=False):
    # Generate the circle for the continuous manifold
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)

    # Create the plot
    plt.figure(figsize=figsize)
    plt.plot(circle_x, circle_y, label='Continuous Manifold (Circumference)', color='gray')
    plt.scatter(data[:, 0], data[:, 1], color='red', label='Train Data (Patterns)')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(fontsize=fontsize)

    # Save the plot
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, filename)

    if show:
        plt.show()
    else:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()

def plot_score2DCircle(grid_tensor, samples, scores, patterns, figsize=(8, 8),
                       fontsize=12, save_path="./",show=False,radius=1):
    # Generate the circle for the continuous manifold
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x = radius * np.cos(theta)
    circle_y = radius * np.sin(theta)

    # Plot
    plt.figure(figsize=figsize)
    plt.grid(False)
    plt.plot(circle_x, circle_y, label='Continuous Manifold', color='blue', alpha=0.25)
    plt.quiver(grid_tensor[:, 0], grid_tensor[:, 1], scores[:, 0], scores[:, 1], width=0.005)
    plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(),   marker="o",  s=10, label="Generated Samples")
    plt.scatter(patterns[:, 0], patterns[:, 1], marker="*",  s=50, color="red", label="Patterns")
    plt.legend(fontsize=fontsize)

    if show:
        plt.show()
    else:
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def visualize_dataset_splits(sample_sizes, seed=9, save_dir=None):
    theta = np.linspace(0, 2 * np.pi, 100)
    circle_x, circle_y = np.cos(theta), np.sin(theta)

    fig, axes = plt.subplots(1, len(sample_sizes), figsize=(5 * len(sample_sizes), 5))
    
    for idx, (ax, sample_size) in enumerate(zip(axes, sample_sizes)):
        ax.plot(circle_x, circle_y, label='Continuous Manifold', color='k', linewidth=2, alpha=0.5)
        train_subset, _ = prepare_datasets(sample_size, seed=seed)
        train_data = train_subset.dataset[train_subset.indices]
        ax.scatter(train_data[:, 0], train_data[:, 1], label='Train Data', s=80, alpha=0.5)
        ax.set_title(f'Sample Size: {sample_size}', fontsize=16)
        if idx == 0:
            ax.legend()
    
    plt.tight_layout()

    if save_dir:
        fig.savefig(os.path.join(save_dir, f'dataset_splits_{seed}.png'), bbox_inches='tight')
        print(f"Dataset splits saved to {os.path.join(save_dir, f'dataset_splits_{seed}.png')}")
    
    plt.show()
    plt.close(fig)
 

def create_grid(bounds=(-1.5, 1.5), resolution=20): 
    x = torch.linspace(bounds[0], bounds[1], resolution)
    y = torch.linspace(bounds[0], bounds[1], resolution)
    X, Y = torch.meshgrid(x, y, indexing='ij')
    return X, Y, torch.stack([X.flatten(), Y.flatten()], dim=1)


def plot_energy_with_scores(grid_points, energy, scores, patterns, save_path=None):
    """"Assumes that the energy and scores are computed on the grid points"""

    # Reshape energy and scores to match the grid points
    h = int(np.sqrt(energy.shape[0])) 
    energy_grid = energy.reshape(h, h)
    scores_grid = scores.reshape(h, h, 2)

    # Generate circle outline points
    circle = plt.Circle((0, 0), 1, color='black', linewidth=2, alpha=1, fill=False)
    
    # Plot
    plt.figure(figsize=(8, 6))
    plt.grid(False)

    # Plot energy contour
    X = grid_points[:,0].reshape(h,h)
    Y = grid_points[:,1].reshape(h,h)
    plt.contourf(X, Y, energy_grid, levels=100, cmap='inferno')
    cbar = plt.colorbar(label='Energy')    
    
    # Plot score field
    plt.quiver(grid_points[:, 0], grid_points[:, 1], 
              scores_grid[:, :, 0], scores_grid[:, :, 1], 
              color='white', alpha=0.7)

    # Plot circle and data points
    plt.gca().add_artist(circle) 
    plt.scatter(patterns[:, 0], patterns[:, 1], marker="*",  s=350, color="red", label="Patterns")

    # Style adjustments
    cbar.set_label('Energy', fontsize=35)
    cbar.ax.yaxis.set_tick_params(labelsize=25)
    plt.gca().set_axis_off()
    plt.tight_layout(pad=0)
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', transparent=True, dpi=200)
    plt.show()


def plot_energy_landscape(energy, scores, grid_tensor, samples, patterns, sample_size,
                         cluster_data=None, show_clusters=True, save_dir=None, 
                         fontsize=12, t=1e-5, reversed=False, legend=False):
    """
    Create a visualization of the energy landscape with samples, scores and optionally clusters.
    """
    grid_points = grid_tensor.detach().cpu().numpy()
    X = grid_tensor[:, 0].reshape(20, 20).detach().cpu().numpy()
    Y = grid_tensor[:, 1].reshape(20, 20).detach().cpu().numpy()
    grid_size = int(np.sqrt(grid_tensor.shape[0]))
    
    # fig = plt.figure(figsize=(8, 6))
    fig = plt.figure(figsize=(6, 6))
    
    # Plot energy contour first
    energy_grid = energy.reshape(grid_size, grid_size).detach().cpu().numpy()
    plt.contourf(X, Y, energy_grid, levels=100, cmap='inferno')
    # cbar = plt.colorbar(label='Energy')
    
    
    # Plot score field
    plt.quiver(grid_points[:, 0], grid_points[:, 1], 
              scores[:, 0], scores[:, 1], 
              color='white', alpha=0.7)

    circle = plt.Circle((0, 0), 1, color='black', linewidth=2, alpha=1, fill=False)
    plt.gca().add_artist(circle)

    if show_clusters == False:
        # Plot patterns
        plt.scatter(patterns[:, 0], patterns[:, 1], 
                marker="*", color="red", s=350, 
                alpha=1, label="Patterns")
    
    if show_clusters and cluster_data is not None:
        cluster_centers, cluster_labels, cluster_energies = cluster_data
        
        if not reversed:  
            # Plot clustered samples with improved visualization
            plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), 
                            c=cluster_labels, marker="o", s=50, 
                            alpha=0.8, cmap='coolwarm', 
                            edgecolor='black', linewidth=0.5,
                            label="Generated Data")
        
        plt.scatter(patterns[:, 0], patterns[:, 1], 
                    marker="*", color="red", s=500, 
                    alpha=1, label="Patterns")
        
        # Plot cluster centers with dark green color instead of yellow
        plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1],
                            marker='X', s=200, color='yellow', 
                            alpha=1, label='Cluster Centers')
        
        if reversed: 
            plt.scatter(samples[:, 0].cpu(), samples[:, 1].cpu(), 
                            c=cluster_labels, marker="o", s=50, 
                            alpha=0.8, cmap='coolwarm', 
                            edgecolor='black', linewidth=0.5,
                            label="Generated Data")
        print("Cluster centers shape: ", cluster_centers.shape)
        # Add energy values as annotations near cluster centers
        if len(cluster_centers) > 50:
            step = len(cluster_centers) // 50
            selected_centers = cluster_centers[::step]
            selected_energies = cluster_energies[::step]
        else:
            selected_centers = cluster_centers
            selected_energies = cluster_energies

        for center, energy_val in zip(selected_centers, selected_energies):
            plt.annotate(f'{energy_val:.2f}', 
                        (center[0], center[1]),
                        xytext=(12,-2), 
                        textcoords='offset points',
                        color='white',
                        fontsize=fontsize)
    
    if legend:
        plt.legend(fontsize=20, loc='upper right')
    plt.gca().set_axis_off()
    plt.tight_layout(pad=0)
    
    if save_dir:
        filename = f"2d_circle_vesde_clustering_energy_{sample_size}_t_{t:.5f}.png"
        save_path = os.path.join(save_dir, filename)
        plt.savefig(save_path, bbox_inches='tight', transparent=True)
    
    plt.show()
    plt.close()
#------------------------------------------------------------------------------------------------
# Model architecture. This is the score-based model used in the paper.
# to train our 2d samples. 


class FourierEmbedding(torch.nn.Module):
    def __init__(self, embed_dim, scale=16):
        super().__init__()
        self.register_buffer('freqs', torch.randn(embed_dim // 2) * scale)

    def forward(self, x):
        x = x.ger((2 * np.pi * self.freqs).to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )
        
    def forward(self, x):
        return x + self.net(x)

class ScoreNet(nn.Module):
    def __init__(
        self, 
        input_dim=2,
        num_layers=4,
        hidden_dim=128,
        embed_dim=128,
        marginal_prob_std=None
    ):
        super().__init__()
        
        self.act = nn.SiLU()
        self.marginal_prob_std = marginal_prob_std

        # Time embedding
        self.time_embed = nn.Sequential(
            FourierEmbedding(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU()
        )

        # Project combined (x + time-embedding) to hidden dimension
        self.input_proj = nn.Linear(input_dim + embed_dim, hidden_dim)

        # Hidden MLP layers
        layers = []
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.SiLU())
        self.hidden = nn.Sequential(*layers)

        # Final output to 2 dimensions
        self.output = nn.Linear(hidden_dim, 2)

    def forward(self, x, t):
        # Generate time embedding and concatenate with x
        t_emb = self.time_embed(t)
        h = torch.cat([x, t_emb], dim=1)
        
        # Pass through MLP
        h = self.input_proj(h)
        h = self.hidden(h)
        h = self.output(h)
        
        # Scale by 1 / marginal_prob_std(t)
        return h / self.marginal_prob_std(t)[:, None]
    

#------------------------------------------------------------------------------------------------
# SDE Terms define the Gaussian kernel used.

class VESDETerms:
    def __init__(self, sigma, device=None):
        self.sigma = sigma
        self.device = device

    def marginal_prob_std(self, t):
        t = torch.as_tensor(t, device=self.device, dtype=torch.float32)
        return self.sigma * torch.sqrt(t)

    def diffusion_coeff(self, t):
        t = torch.as_tensor(t, device=self.device, dtype=torch.float32)
        return self.sigma * torch.ones_like(t)

#------------------------------------------------------------------------------------------------
# Sampler 

def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=1000, 
                           device='cuda', 
                           eps=1e-3):
  """Generate samples from score-based models with the Euler-Maruyama solver."""
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 2, device=device) \
    * marginal_prob_std(t)[:, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in tqdm(time_steps):      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2)[:, None] * score_model(x, batch_time_step) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x)      
  # Do not include any noise in the last sampling step.
  return mean_x

#------------------------------------------------------------------------------------------------
# Training utils

def score_eval_wrapper(sample, time_steps, score_model):
    """A wrapper of the score-based model for use by the ODE solver."""
    # Keep original shape for reshaping later
    original_shape = sample.shape 
    device = sample.device    
    # Convert to tensor if not already
    if not isinstance(sample, torch.Tensor):
        sample = torch.tensor(sample, device=device, dtype=torch.float32)
    sample = sample.reshape(original_shape)
    
    if not isinstance(time_steps, torch.Tensor):
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32)
    time_steps = time_steps.reshape((sample.shape[0], ))    
    
    with torch.no_grad():    
        score = score_model(sample, time_steps)
    # Reshape score to match expected 2D shape for plotting
    return score.cpu().numpy().reshape((-1, 2)).astype(np.float64)


def loss_fn(model, x, marginal_prob_std, eps=1e-5):
  random_t = torch.rand(x.shape[0], device=x.device) * (1. - eps) + eps  
  z = torch.randn_like(x)
  std = marginal_prob_std(random_t)
  perturbed_x = x + z * std[:, None] 
  score = model(perturbed_x, random_t)
  loss = torch.mean(torch.sum((score * std[:, None] + z)**2, dim=1))
  return loss 


def train_score_model(score_model, train_loader, test_loader, optimizer, device, 
                     marginal_prob_std_fn, diffusion_coeff_fn, total_iterations=100000,
                     initial_step=0, eps=1e-12, sampling_dir=None, log_freq=100,
                     model_save_dir=None, 
                     sampling_freq=1000): 
    dataset_name = "circle"
    # Generation setup for visualization
    X, Y, grid_points = create_grid(bounds=(-1.5, 1.5), resolution=20)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=-1)
    grid_tensor = torch.tensor(grid_points, dtype=torch.float32)

    # Training monitoring variables
    log_steps = 0
    running_loss = 0
    losses = [] 
    eval_losses = []
    print("Total iterations: ", total_iterations)
    infinite_loader = iter(cycle(train_loader))
    
    for iteration in range(initial_step, total_iterations + 1): 
            
        # Get batch
        x = next(infinite_loader)
        x = x.to(device).type(torch.float32)
        
        # Compute loss
        loss = loss_fn(score_model, x, marginal_prob_std_fn, eps=eps)
        
        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        log_steps += 1
        losses.append(loss.item())
        
        if iteration % log_freq == 0 and iteration > 0:
            avg_loss = running_loss / log_steps
            print(f"Iteration {iteration}/{total_iterations}, Avg Loss : {avg_loss:.6f}")
            
            # Compute evaluation loss
            eval_loss = 0.0
            score_model.eval()  # Set the model to evaluation mode for eval loss
            with torch.no_grad():
                for test_x in test_loader:
                    test_x = test_x.to(device).type(torch.float32)
                    eval_loss += loss_fn(score_model, test_x, marginal_prob_std_fn, eps=eps).item()
                eval_loss /= len(test_loader)
            eval_losses.append(eval_loss)
            score_model.train()  # Set the model back to training mode 
            
        # Sample and visualize
        if iteration % sampling_freq == 0 and iteration > 0:
            score_model.eval()  # Set the model to evaluation mode for sampling
            t = 1e-5
            vec_t = torch.ones(grid_tensor.shape[0], device=device) * t
            scores = score_eval_wrapper(grid_tensor.to(device), vec_t, score_model)
            samples = Euler_Maruyama_sampler(score_model, 
                            marginal_prob_std_fn,
                            diffusion_coeff_fn, 
                            1000,
                            device=device)

            name = "{}_{}_samples_{}.png".format("vesde", dataset_name, iteration)
            if sampling_dir is not None:
                save_path = os.path.join(sampling_dir, name)            
                plot_score2DCircle(grid_tensor, samples.detach().cpu(), scores, x.cpu(),
                                  figsize=(6, 6), fontsize=8, save_path=save_path)
            
            score_model.train()  
    
    if model_save_dir is not None:
        model_save_path = os.path.join(model_save_dir,  f'ckpt_{iteration}.pth')
        torch.save(score_model.state_dict(), model_save_path)
        print(f"Model saved at iteration {iteration} to {model_save_path}")

    # Plot final training and evaluation loss curves:
    window_size = 100
    plt.figure(figsize=(12, 6))
    plt.plot(losses[1:], 'b-', alpha=0.3, label='Training Loss')
    plt.plot(range(log_freq, total_iterations + 1, log_freq), eval_losses, 'g-', label='Evaluation Loss')
    
    if len(losses[1:]) >= window_size:
        moving_avg = np.convolve(losses[1:], np.ones(window_size)/window_size, mode='valid')
        plt.plot(range(window_size-1, len(losses)-1), moving_avg, 'r-',
                label=f'Moving Average (window={window_size})')
        
    plt.xlabel('Iteration')
    plt.ylabel('Loss') 
    plt.yscale('log')
    plt.grid(True)
    plt.legend(loc='upper right')
    plt.title('Training and Evaluation Loss Evolution')
    plt.savefig(os.path.join(sampling_dir, 'loss_evolution.png'))
    plt.show()
        
    return score_model

#------------------------------------------------------------------------------------------------

def train_model(sample_size=2,
                n_iter=100,  
                sampling_freq=1000,
                batch_size=500,
                sigma=1.0,
                eps=1e-5,
                ):
    print("Training model over eps={}".format(eps))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Prepare datasets
    train_subset, test_subset = prepare_datasets(sample_size) 
    train_loader = DataLoader(train_subset, batch_size=min(batch_size, sample_size), shuffle=True)
    test_loader = DataLoader(test_subset, batch_size=min(batch_size, sample_size), shuffle=True)
    patterns = next(iter(train_loader))
    print(patterns.shape)
    # Create results directory based on sample_size
    results_dir = f'./results/toy_example_results/sample_size_{sample_size}'
    sample_dir = os.path.join(results_dir, "samples")   
    model_dir = os.path.join(results_dir, "models") 
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    save_circle_plot(
        patterns,
        save_dir=results_dir,
        filename=f"2dcircle_data_samples",
        figsize=(3, 3),
        fontsize=6,
        show=False
    )   

    # Define SDE and related functions
    vesde = VESDETerms(sigma, device)
    marginal_prob_std_fn = vesde.marginal_prob_std
    diffusion_coeff_fn = vesde.diffusion_coeff

    # Define score model
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
    score_model = score_model.to(device)
    optimizer = torch.optim.Adam(score_model.parameters(), lr=1e-4)

    # Train the score model
    score_model = train_score_model(
        score_model=score_model,
        train_loader=train_loader,
        test_loader=test_loader,
        optimizer=optimizer, 
        total_iterations=n_iter,
        device=device,
        marginal_prob_std_fn=marginal_prob_std_fn,
        diffusion_coeff_fn=diffusion_coeff_fn,
        eps=eps,   
        sampling_dir=sample_dir,
        model_save_dir=model_dir,  
        sampling_freq=sampling_freq
    )

#------------------------------------------------------------------------------------------------
# Likelihood Computation as described in Song et al. 2021 using the Laplacian instead of the Hutchinson trace estimator

def prior_likelihood(z, sigma):
  """The likelihood of a Gaussian distribution with mean zero and 
      standard deviation sigma."""
  shape = z.shape
  N = np.prod(shape[1:])
  return -N / 2. * torch.log(2*np.pi*sigma**2) - torch.sum(z**2, dim=1) / (2 * sigma**2)


def compute_laplacian(score_fn, x, t):
    """Compute the Laplacian of the score function."""
    laplacian = torch.zeros(x.size(0), device=x.device)
    with torch.enable_grad():
        x.requires_grad_(True)
        score = score_fn(x, t)
        for i in range(x.shape[1]):  # Iterate over dimensions
            grad_score_i = torch.autograd.grad(score[:, i].sum(), x, create_graph=True)[0][:, i]
            laplacian += grad_score_i
    x.requires_grad_(False)  # Make sure to disable gradient tracking after computation
    return laplacian.detach()  # Detach the tensor to avoid gradient tracking issues


def ode_likelihood_with_laplacian(x, 
                                  score_model,
                                  marginal_prob_std, 
                                  diffusion_coeff,
                                  batch_size=64, 
                                  device='cuda',
                                  eps=1e-5):
    """Compute the likelihood with probability flow ODE using explicit Laplacian."""
    shape = x.shape

    def score_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the score-based model for the ODE solver."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
        with torch.no_grad():    
            score = score_model(sample, time_steps)
        return score.cpu().numpy().reshape((-1, shape[1])).astype(np.float64)

    def laplacian_eval_wrapper(sample, time_steps):
        """A wrapper for evaluating the Laplacian of the score function."""
        sample = torch.tensor(sample, device=device, dtype=torch.float32).reshape(shape)
        time_steps = torch.tensor(time_steps, device=device, dtype=torch.float32).reshape((sample.shape[0], ))
        laplacian = compute_laplacian(score_model, sample, time_steps)
        return laplacian.cpu().numpy().reshape((-1,)).astype(np.float64)

    def ode_func(t, x):
        """The ODE function for the solver."""
        time_steps = np.ones((shape[0],)) * t
        sample = x[:-shape[0]] 
        g = diffusion_coeff(torch.tensor(t)).cpu().numpy()
        drift = -0.5 * g**2 * score_eval_wrapper(sample, time_steps)
        logp_grad = -0.5 * g**2 * laplacian_eval_wrapper(sample, time_steps)
        return np.concatenate([drift.reshape(-1), logp_grad], axis=0)

    init = np.concatenate([x.cpu().numpy().reshape((-1,)), np.zeros((shape[0],))], axis=0)
    # Black-box ODE solver
    res = integrate.solve_ivp(ode_func, (eps, 1.), init, rtol=1e-5, atol=1e-5, method='RK45')
    zp = torch.tensor(res.y[:, -1], device=device)
    z = zp[:-shape[0]].reshape(shape)
    delta_logp = zp[-shape[0]:].reshape(shape[0])
    sigma_max = marginal_prob_std(1.)
    prior_logp = prior_likelihood(z, sigma_max)
    logp = prior_logp + delta_logp
    return logp



#------------------------------------------------------------------------------------------------
# Energy
def potential_energy(logp, t=1e-5, std=1):   
    energy = -   logp * (2 * t * std*2 )
    return energy

#------------------------------------------------------------------------------------------------
#  Utils for energy visualizaiton 
def compute_clusters(samples, distance_threshold=1.0):
    """
    Compute clusters from samples using hierarchical clustering.
    """
    # Convert samples to numpy if needed
    samples_np = samples.cpu().numpy() if torch.is_tensor(samples) else samples
    
    # Perform hierarchical clustering
    clustering = AgglomerativeClustering(
        n_clusters=None, 
        distance_threshold=distance_threshold
    ).fit(samples_np)
    
    # Compute cluster centers
    unique_labels = np.unique(clustering.labels_)
    cluster_centers = np.array([
        samples_np[clustering.labels_ == label].mean(axis=0) 
        for label in unique_labels
    ])
    
    return cluster_centers, clustering.labels_


def dbscan_clustering(samples, eps=0.5, min_samples=5):
    """
    Perform density-based clustering (DBSCAN).
    
    Parameters:
        samples (numpy.ndarray or torch.Tensor): Input data points.
        eps (float): Maximum distance between points in a cluster.
        min_samples (int): Minimum points needed to form a cluster.
    
    Returns:
        tuple: (cluster_centers, cluster_labels)
    """
    samples_np = samples.cpu().numpy() if torch.is_tensor(samples) else samples
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(samples_np)
    
    labels = clustering.labels_
    unique_labels = np.unique(labels[labels >= 0])  # Ignore noise (-1)

    # Compute cluster centers
    cluster_centers = np.array([
        samples_np[labels == label].mean(axis=0)
        for label in unique_labels
    ])

    return cluster_centers, labels


def kmeans_clustering(samples, n_clusters=2):
    """
    Perform K-Means clustering.
    
    Parameters:
        samples (numpy.ndarray or torch.Tensor): Input data points.
        n_clusters (int): Number of clusters.
    
    Returns:
        tuple: (cluster_centers, cluster_labels)
    """
    samples_np = samples.cpu().numpy() if torch.is_tensor(samples) else samples
    clustering = KMeans(n_clusters=n_clusters, random_state=42).fit(samples_np)
    
    return clustering.cluster_centers_, clustering.labels_


def get_optimal_distance_threshold(distances, samples, sample_size, save_dir=None):
    """
    Dynamically finds the optimal distance threshold by identifying the valley (local minimum)
    between peaks in the pairwise distance distribution.

    Parameters:
        distances (numpy.ndarray): A 1D array of pairwise distances.
        plot (bool): If True, visualizes the distance distribution and threshold.

    Returns:
        float: The optimal distance threshold for clustering.
    """
    #  Compute pairwise distances
    flat_distances = distances[np.triu_indices(len(samples), k=1)]

    # Estimate density to find peaks and valleys
    density = gaussian_kde(flat_distances)
    x_vals = np.linspace(flat_distances.min(), flat_distances.max(), 500)
    density_vals = density(x_vals)

    # Find peaks (clusters) and valleys (potential thresholds)
    peaks, _ = find_peaks(density_vals)
    valleys, _ = find_peaks(-density_vals)  # Negative peaks = valleys

    # Count peaks that have at least 1 or more elements
    num_valid_peaks = sum(density_vals[peaks] > 0) + sum(density_vals[valleys] > 0)
    print("Number of valid peaks: ", num_valid_peaks)
    # Choose the first valley between the two highest peaks as the threshold
    if len(valleys) > 0:
        distance_threshold = x_vals[valleys[0]]  # First valley
    else:
        distance_threshold = np.median(flat_distances)  # Fallback

    # Optional: Plot the histogram with the detected threshold
    if save_dir is not None:
        plt.figure(figsize=(8, 5))
        plt.hist(flat_distances, bins=50, alpha=0.5, density=True, label="Pairwise Distances")
        plt.plot(x_vals, density_vals, label="Density Estimate", color='blue')

        # Mark peaks and valleys
        plt.scatter(x_vals[peaks], density_vals[peaks], color='red', label="Peaks", zorder=3)
        plt.scatter(x_vals[valleys], density_vals[valleys], color='green', label="Valleys", zorder=3)
        plt.axvline(distance_threshold, color='black', linestyle='--', label=f"Threshold: {distance_threshold:.2f}")

        plt.xlabel("Pairwise Distance")
        plt.ylabel("Density") 
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'distance_threshold_sample_size_{}.png'.format(sample_size))) 
    return distance_threshold, num_valid_peaks

#------------------------------------------------------------------------------------------------
# Evaluate the model
def load_model(model_dir, checkpoint, sigma=1.0): 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Define SDE terms
    vesde = VESDETerms(sigma, device)
    marginal_prob_std_fn = vesde.marginal_prob_std
    diffusion_coeff_fn = vesde.diffusion_coeff

    # Initialize the model
    score_model = ScoreNet(marginal_prob_std=marginal_prob_std_fn)
    score_model = score_model.to(device)

    # Load the model state_dict
    model_path = os.path.join(model_dir, f"ckpt_{checkpoint}.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}")
    score_model.load_state_dict(torch.load(model_path, map_location=device))
    score_model.eval()

    return score_model, marginal_prob_std_fn, diffusion_coeff_fn


def evaluate_model(sample_size=2, t=0.15, checkpoint=500000, batch_size=10000, distance_threshold=1.0, dynamic_threshold=False):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    workdir = f'./results/toy_example_results/sample_size_{sample_size}'
    model_dir = os.path.join(workdir, "models")
    sample_dir = os.path.join(workdir, "energy_plots") 
    os.makedirs(sample_dir, exist_ok=True) 

    # Load Data 
    train_subset, _ = prepare_datasets(sample_size) 
    train_loader = DataLoader(train_subset, batch_size=sample_size, shuffle=True) 
    patterns = next(iter(train_loader))

    # Load Model
    score_model, marginal_prob_std_fn, diffusion_coeff_fn = load_model(model_dir, checkpoint)

    # Sample from the model
    samples = Euler_Maruyama_sampler(score_model, 
                      marginal_prob_std_fn,
                      diffusion_coeff_fn, 
                      batch_size,
                      device=device)
    if dynamic_threshold:
        print("Finding the optimal threshold dynamically")
        # Compute pairwise distances
        distances = distance_matrix(samples.cpu().numpy(), samples.cpu().numpy())
        # Find the optimal threshold
        distance_threshold, num_valid_peaks =  get_optimal_distance_threshold(distances, samples, sample_size, save_dir=sample_dir)
        cluster_centers, cluster_labels = dbscan_clustering(samples, eps=distance_threshold, min_samples=2)
    else:
        distance_threshold = distance_threshold
        print("Using the provided distance threshold: ", distance_threshold)
        cluster_centers, cluster_labels = compute_clusters(samples, distance_threshold=distance_threshold)
        # cluster_centers, cluster_labels =  kmeans_clustering(samples, n_clusters=num_valid_peaks)
    # ------------------------------------------------------------------------------------------------
    print("Model evaluation with distance threshold: ", distance_threshold)

    # Create a grid of points for evaluation
    grid_tensor = create_grid(resolution=20)[2].clone().detach().to(torch.float32)


    # Gets the likelihood at t_0 with ode_likelihood_with_laplacian
    logp = ode_likelihood_with_laplacian(grid_tensor.to(device),
                                                score_model,
                                                marginal_prob_std_fn,
                                                diffusion_coeff_fn,
                                                grid_tensor.shape[0],
                                                device=device,
                                                eps=t)
    # Compute the energy at a given time
    vec_t = torch.ones(grid_tensor.shape[0], device=device) * t
    energy = potential_energy(logp, t=t) 
    normalized_energy = energy - energy.min()
    scores = score_eval_wrapper(grid_tensor.to(device), vec_t, score_model)

    

    lopg_clusters = ode_likelihood_with_laplacian(torch.tensor(cluster_centers, device=device),
                                                            score_model,
                                                            marginal_prob_std_fn,
                                                            diffusion_coeff_fn,
                                                            device=device,
                                                            eps=t)
    cluster_energies = potential_energy(lopg_clusters, t)
    cluster_energies = cluster_energies - cluster_energies.min()
    # cluster_energies = torch.round(potential_energy(lopg_clusters, t_energy) * 100) / 100
    cluster_data = (cluster_centers, cluster_labels, cluster_energies)

    plot_energy_landscape(normalized_energy,
                        scores,
                        grid_tensor,
                        samples,
                        patterns,
                        sample_size,
                        cluster_data=cluster_data,
                        save_dir=sample_dir, 
                        fontsize=14,
                        reversed=False,  
                        legend=False)

#------------------------------------------------------------------------------------------------
# Compute the Basin of attraction 

def inject_noise(x0, t, marginal_prob_std_fn, device="cpu"):
    # Get a noisy version of the patterns for a given time step 
    t_vec = torch.full((x0.shape[0],), t, device=device) 
    z = torch.randn_like(x0, device=device)
    std = marginal_prob_std_fn(t_vec).to(device)
    perturbed_x = x0 + z * std[:, None] 
    return perturbed_x


def Euler_Maruyama_sampler(score_model, 
                           marginal_prob_std,
                           diffusion_coeff, 
                           batch_size=64, 
                           num_steps=1000, 
                           device='cuda', 
                           eps=1e-3):
  """Generate samples from score-based models with the Euler-Maruyama solver."""
  t = torch.ones(batch_size, device=device)
  init_x = torch.randn(batch_size, 2, device=device) \
    * marginal_prob_std(t)[:, None]
  time_steps = torch.linspace(1., eps, num_steps, device=device)
  step_size = time_steps[0] - time_steps[1]
  x = init_x
  with torch.no_grad():
    for time_step in tqdm(time_steps):      
      batch_time_step = torch.ones(batch_size, device=device) * time_step
      g = diffusion_coeff(batch_time_step)
      mean_x = x + (g**2)[:, None] * score_model(x, batch_time_step) * step_size
      x = mean_x + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x)      
  # Do not include any noise in the last sampling step.
  return mean_x


@torch.no_grad()
def reverse_diffusion_sample(score_model, x_t, t, diffusion_coeff_fn, device='cpu', eps=1e-3):
    batch_size = x_t.size(0)
    num_steps = int(1000 * t) 
    time_steps = torch.linspace(t, eps, num_steps, device=device, dtype=torch.float32)  # Set dtype to float32
    step_size = time_steps[0] - time_steps[1]
    x = x_t.clone().to(device, dtype=torch.float32)  # Ensure dtype consistency

    for time_step in tqdm(time_steps, desc="Reverse diffusion"):
        batch_time_step = torch.ones(batch_size, device=device, dtype=torch.float32) * time_step
        g = diffusion_coeff_fn(batch_time_step).to(dtype=torch.float32)  # Convert to float32
        drift = (g**2)[:, None] * score_model(x, batch_time_step) * step_size
        x_mean = x + drift
        x = x_mean + torch.sqrt(step_size) * g[:, None] * torch.randn_like(x, dtype=torch.float32)
    
    return x_mean


def euclidean_distance(x, y):
    """ Returns ||x - y||_2 per sample. x, y shape: (B,2). """
    return torch.norm(x - y, dim=-1)


# Simulation: Inject noise at time t and reverse back
def simulate_noise_injection_and_reversal(patterns, score_model, diffusion_coeff_fn, marginal_prob_std_fn, t=0.5, device='cpu'):
    patterns = patterns.to(device)
    noisy_patterns = inject_noise(patterns, t, marginal_prob_std_fn, device=device)
    reversed_patterns = reverse_diffusion_sample(score_model.to(device), noisy_patterns, t, 
                                                 diffusion_coeff_fn, device=device)
    return patterns, noisy_patterns, reversed_patterns

# Plot the original, noisy, and reversed patterns
def plot_noise_injection_and_reversal(patterns, reversed_patterns):
    plt.figure(figsize=(12, 6))
    plt.scatter(patterns[:, 0].cpu(), patterns[:, 1].cpu(), color='blue', label='Original Patterns')
    # plt.scatter(noisy_patterns[:, 0].cpu(), noisy_patterns[:, 1].cpu(), color='red', label='Noisy Patterns')
    plt.scatter(reversed_patterns[:, 0].cpu(), reversed_patterns[:, 1].cpu(), color='green', label='Reversed Patterns')
    plt.legend()
    plt.title('Noise Injection and Reversal Simulation')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(-2, 2)  # Fix x-axis limits
    plt.ylim(-2, 2)  # Fix y-axis limits
    plt.gca().set_aspect('equal', adjustable='box')
    plt.show()


def log_volume_hypersphere(radius, dimension):
    """
    Compute the logarithm of the volume of a hypersphere given the radius and dimensionality
    :param radius: radius of the hypersphere
    :param dimension: dimensionality of the hypersphere
    :return: log (base e) of the volume of the hypersphere
    """
    if radius <= 0 or dimension <= 0:
        raise ValueError("Radius and dimensionality should be positive")
    log_num = (dimension / 2) * math.log(math.pi * (radius ** 2))
    log_denom = math.lgamma(dimension / 2 + 1)
    log_volume = log_num - log_denom

    return log_volume


def find_optimal_time(pattern, score_model, diffusion_coeff_fn, marginal_prob_std_fn, device='cpu', p_trials=10, delta=0.1, p_threshold=0.9):
    """Finds the optimal noise injection time for a given pattern."""
    pattern = pattern.to(device)  # Ensure it's on the correct device
    t_values = np.linspace(1e-2, 1.0, 500)  # Test times from 0.1 to 1.0

    success_for_t = []
    t_opt = 0
    for tval in t_values:
        # Repeat pattern for multiple trials
        x0 = pattern.repeat(p_trials, 1)  

        # Inject noise and reverse diffusion
        _, x_t, x_0_hat = simulate_noise_injection_and_reversal(
            x0, score_model, diffusion_coeff_fn, marginal_prob_std_fn, tval, device=device
        )

        # Plot every 5th pattern to avoid excessive plots
        # if i % 5 == 0:
        #     plot_noise_injection_and_reversal(x0, x_0_hat)

        # Compute recovery distances
        dist = euclidean_distance(x_0_hat, pattern)  
        recovered = (dist <= delta).float()  

        # Compute success probability
        fraction_ok = recovered.mean().item()
        success_for_t.append(fraction_ok)

        # Stop early if recovery probability drops
        if fraction_ok >= p_threshold:
            t_opt = tval
        else:
            break  # Exit loop when recovery fails

    return t_opt, success_for_t


def compute_optimal_recovery_time(sample_size=2, checkpoint=500000, device='cpu', delta=0.05):
    """Compute the optimal recovery time for a given sample size and checkpoint"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    workdir = f'./results/toy_example_results/sample_size_{sample_size}'
    model_dir = os.path.join(workdir, "models") 
    results_dir = './results/toy_example_results/volume_of_basin_of_attraction'
    os.makedirs(results_dir, exist_ok=True) 

    # Load Data 
    train_subset, _ = prepare_datasets(sample_size) 
    train_loader = DataLoader(train_subset, batch_size=sample_size, shuffle=True) 
    patterns = next(iter(train_loader)) 
    patterns = patterns.to(device)

    # Load Model
    score_model, marginal_prob_std_fn , diffusion_coeff_fn = load_model(model_dir, checkpoint)

    times = []
    radii = [] 

    for i, x0 in enumerate(patterns):
        # Find the best recovery time for this pattern
        opt_time, _ = find_optimal_time(x0, score_model, diffusion_coeff_fn, marginal_prob_std_fn, device=device, delta=delta)

        # Inject noise at the optimal recovery time
        x_t = inject_noise(x0, opt_time, marginal_prob_std_fn, device=device)  
        dist = torch.norm(x_t - x0).cpu().numpy()

        times.append(opt_time)  # Store optimal recovery time
        radii.append(dist)  # Store recovery radius
        print(f"Pattern {i}: Optimal Time = {opt_time}, Radius = {dist}")

    # Save the results
    save_path = os.path.join(results_dir, f'volume_of_basin_of_attraction_sample_size_{sample_size}.npz')
    np.savez_compressed(save_path, time=times, radius=radii, delta=delta)

    return times, radii


def load_volume_of_basin_of_attraction():
    """Load the volume of basin of attraction for a given sample size and checkpoint"""
    results_dir = './results/toy_example_results/volume_of_basin_of_attraction' 
    times = []
    radii = []
    for sample_size in [2, 9, 1000]:
        save_path = os.path.join(results_dir, f'volume_of_basin_of_attraction_sample_size_{sample_size}.npz')
        data = np.load(save_path)
        times = data['time']
        radii = data['radius']
        return times, radii
    
def compute_basin_of_attraction(delta=0.5):
    """Compute the basin of attraction for a given sample size and checkpoint"""
    results_dir = './results/toy_example_results/volume_of_basin_of_attraction' 
    for sample_size in [2]:
        times, radii = compute_optimal_recovery_time(sample_size=sample_size, checkpoint=500000, delta=delta)
        # log_volumes = []
        # for time, radius in zip(times, radii):
        #     log_volume = log_volume_hypersphere(radius, sample_size)
        #     log_volumes.append(log_volume)
        # plt.figure(figsize=(8, 6))
        # plt.scatter(times, log_volumes, alpha=0.7, color='blue', label="Recovery Points")
    

#------------------------------------------------------------------------------------------------
# Main Command Line 

@click.group()
def cmdline():
    """2D toy example from the paper : "Memorization to Generalization: Diffusion Models from Dense Associative Memory"
    
    Examples:

    \b
    # Visualize the data for different sample sizes given the choosen seed. Useful to choose the seed based on 
    # how the data is spread on the circle.
    python toy_example.py data --seed 9

    \b
    # Generate the exact energy and score plots for a given beta
    python toy_example.py exact --sample_size 2 --beta 20.0

    \b
    # Train the model
    python toy_example.py train --sample_size 2 --n_iter 500000 --sampling_freq 50000


    \b 
    # Generate the energy and score plots for a given trained diffusion model 
    python toy_example.py plots --sample_size 2 --t 0.25 --checkpoint 800000 --distance_threshold 3.0 --batch_size 10000
    python toy_example.py plots --sample_size 9 --t 0.15 --checkpoint 800000 --distance_threshold 2. --batch_size 10000 
    python toy_example.py plots --sample_size 1000 --t 0.15 --checkpoint 500000 --distance_threshold 0.45 --batch_size 10000  

    python toy_example.py plots --sample_size 2 --t 0.05 --checkpoint 800000 --distance_threshold 3.0 --batch_size 10000
    python toy_example.py plots --sample_size 9 --t 0.05 --checkpoint 800000 --distance_threshold 2. --batch_size 10000 
    python toy_example.py plots --sample_size 1000 --t 0.05 --checkpoint 500000 --distance_threshold 0.45 --batch_size 10000

    \b
    # Use DBSCAN clustering
    python toy_example.py plots --sample_size 2 --t 0.15 --checkpoint 500000 --batch_size 10000 --dynamic_threshold True
    python toy_example.py plots --sample_size 9 --t 0.15 --checkpoint 500000 --batch_size 1000 --dynamic_threshold True
    python toy_example.py plots --sample_size 1000 --t 0.15 --checkpoint 500000 --distance_threshold 1.0 --batch_size 10000 --dynamic_threshold True
    
    \b
    # Compute the optimal recovery time
    python toy_example.py basin --sample_size 2 --checkpoint 500000 --delta 0.01
    python toy_example.py basin --sample_size 9 --checkpoint 500000 --delta 0.01
    python toy_example.py basin --sample_size 1000 --checkpoint 500000 --delta 0.01
    """

#------------------------------------------------------------------------------------------------
# Subcommands Visualize Data splits
@cmdline.command()
@click.option('--seed',             help='Seed for the data', metavar='INT', type=int, default=9, show_default=True)
def data(seed):
    """Visualize the data splits"""
    results_dir = os.path.join('results', 'toy_example_results')
    os.makedirs(results_dir, exist_ok=True) 
    sample_size = [2, 4, 9, 1000]
    visualize_dataset_splits(sample_size, seed, save_dir=results_dir)

#------------------------------------------------------------------------------------------------
# Subcommands Exact Energy and Score
@cmdline.command()
@click.option('--sample_size',      help='Number of patterns to display', metavar='INT',     type=int, default=1000, show_default=True)
@click.option('--beta',             help='Beta value for the energy and score computation', metavar='FLOAT', type=float, default=20.0, show_default=True)
def exact(sample_size, beta):
    """Compute the exact energy and score for a given beta"""
    # Create dataset object and create a subset of the data
    dataset = CircleDataset(num_samples=60000, seed=9) 
    train_subset = create_subset(dataset, sample_size)

    # Create DataLoader
    data_loader = DataLoader(train_subset, batch_size=sample_size, shuffle=True) 
    
    # Get patterns
    patterns = next(iter(data_loader))

    # Create evaluation grid and compute energy and scores for a given beta
    _,_, grid_points = create_grid(resolution=20)
    energy = dataset.energy_am(grid_points, beta)
    scores = dataset.score_am(grid_points, beta)
    
    # Create results directory if it doesn't exist
    results_dir = os.path.join('results', 'toy_example_results')
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot and save the image
    plot_energy_with_scores(grid_points, energy, scores, patterns, save_path=os.path.join(results_dir, 'energy_scores_plot.png'))

#------------------------------------------------------------------------------------------------
# Subcommand Train Model
@cmdline.command()
@click.option('--sample_size',      help='Number of patterns to display',     metavar='INT',     type=int, default=1000, show_default=True)
@click.option('--n_iter',           help='Number of iterations for training', metavar='INT',     type=int, default=800000, show_default=True)
@click.option('--sampling_freq',    help='Frequency of sampling',             metavar='INT',     type=int, default=50000, show_default=True)
def train(sample_size, n_iter, sampling_freq):
    """Train the model"""
    print("Training score model and sampling every {} iterations".format(sampling_freq))
    train_model(sample_size, n_iter, sampling_freq)    


#------------------------------------------------------------------------------------------------
# Subcommand Evaluate Energy Landscape for a given trained Diffusion Model
@cmdline.command()
@click.option('--sample_size',      help='Number of patterns to display', metavar='INT',     type=int, default=1000, show_default=True)
@click.option('--t',                help='Time step for energy landscape', metavar='FLOAT', type=float, default=0.15, show_default=True)
@click.option('--checkpoint',       help='Checkpoint to load', metavar='INT', type=int, default=500000, show_default=True)
@click.option('--distance_threshold', help='Distance threshold for clustering', metavar='FLOAT', type=float, default=1.0, show_default=True)
@click.option('--batch_size',       help='Batch size for sampling', metavar='INT', type=int, default=10000, show_default=True)
@click.option('--dynamic_threshold', help='Use dynamic threshold for clustering', is_flag=False, default=False)
def plots(sample_size, t, checkpoint, distance_threshold, batch_size, dynamic_threshold):
    """Evaluate the energy landscape for a given trained diffusion model"""
    print("Evaluating energy landscape for sample size: ", sample_size, "with distance threshold: ", distance_threshold, "and batch size: ", batch_size)
    evaluate_model(sample_size, t, checkpoint, batch_size, distance_threshold, dynamic_threshold)


#------------------------------------------------------------------------------------------------
# Subcommand Compute Optimal Recovery Time
@cmdline.command()
# @click.option('--sample_size',      help='Number of patterns to display', metavar='INT',     type=int, default=2, show_default=True)
# @click.option('--checkpoint',       help='Checkpoint to load', metavar='INT', type=int, default=500000, show_default=True)
# @click.option('--delta',            help='Distance threshold for recovery', metavar='FLOAT', type=float, default=0.01, show_default=True)
def basin():
    """Compute the optimal recovery time for a given sample size and checkpoint"""
    # compute_optimal_recovery_time(sample_size, checkpoint, delta)
    compute_basin_of_attraction()

if __name__ == '__main__':
    cmdline()