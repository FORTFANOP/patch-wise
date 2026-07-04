import random
import matplotlib.pyplot as plt

def random_patcher(img):
    x1, x2, y1, y2 = random.sample(range(1, 101), 4)
    # testing

def plot_reconstructions(x_corrupted, reconstructed, x_clean):
    """
    Plots a grid of inputs, reconstructions, and ground truths
    Inputs must be CPU tensors
    """
    batch_size = x_corrupted.shape[0]
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))

    if batch_size == 1:
        axes = [axes]

    for i in range(batch_size):
        # Input
        axes[i][0].imshow(x_corrupted[i].permute(1, 2, 0))
        axes[i][0].set_title("Input")

        # Reconstruction
        axes[i][1].imshow(reconstructed[i].permute(1, 2, 0).detach().numpy())
        axes[i][1].set_title("Reconstruction")

        # Ground Truth
        axes[i][2].imshow(x_clean[i].permute(1, 2, 0))
        axes[i][2].set_title("Ground Truth")

        for j in range(3):
            axes[i][j].axis("off")

    plt.tight_layout()
    plt.show()