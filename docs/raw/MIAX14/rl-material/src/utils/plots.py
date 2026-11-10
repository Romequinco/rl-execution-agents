import matplotlib.pyplot as plt
import numpy as np


def plot_training_results(
    rewards: list,
    losses: list,
    window: int = 100,
    scale: str = "log"
):
    rew_plot = np.convolve(rewards, np.ones(window)/window, mode='valid')
    loss_plot = np.convolve(losses, np.ones(window)/window, mode='valid')
    _, ax = plt.subplots()
    ax_color = "royalblue"
    ax.plot(rew_plot, color=ax_color)
    ax.set_yscale(scale)
    ax.set_ylabel("Reward", color=ax_color)
    ax2 = ax.twinx()
    ax2_color = "lavender"
    ax2.plot(loss_plot, color=ax2_color)
    ax2.set_yscale(scale)
    ax2.set_ylabel("Loss", color=ax2_color)
    ax.set_xlabel("Episodes")
