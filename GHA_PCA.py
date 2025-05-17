import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize

def load_image_blocks(path, block_size=8, target_shape=(256, 256)):
    image = plt.imread(path)
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    if image.shape != target_shape:
        image = resize(image, target_shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)

    height, width = image.shape
    blocks = []
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                blocks.append(block.flatten())
    return np.array(blocks), image.shape, image


def rebuild_image_from_blocks(blocks, shape, block_size=8):
    image = np.zeros(shape, dtype=np.uint8)
    index = 0
    for i in range(0, shape[0], block_size):
        for j in range(0, shape[1], block_size):
            block = blocks[index].reshape((block_size, block_size))
            image[i:i+block_size, j:j+block_size] = np.clip(block, 0, 255)
            index += 1
    return image


def compute_mse(original_img, reconstructed_img):
    diff = original_img.astype(np.float32) - reconstructed_img.astype(np.float32)
    return np.sum(diff ** 2) / (original_img.shape[0] * original_img.shape[1])


class GHA_PCA:
    def __init__(self, num_components=16, alpha_s=1e-3, alpha_f=1e-5, epochs=100):
        self.num_components = num_components  # Number of principal components to learn
        self.alpha_s = alpha_s  # Starting learning rate
        self.alpha_f = alpha_f  # Final learning rate
        self.epochs = epochs
        self.W = None  # Weight matrix = principal components
        self.mean = None  # Mean for normalization
        self.std = None   # Std deviation for normalization

    def fit(self, X):
        # Normalize data (zero mean, unit variance)
        n_samples, n_features = X.shape
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0) + 1e-5
        X_norm = (X - self.mean) / self.std

        # Initialize weights with small values and mean zero
        W = np.random.randn(self.num_components, n_features) * 0.01
        W -= np.mean(W, axis=1, keepdims=True)

        for ep in range(self.epochs):
            # Geometric learning rate schedule
            lr = self.alpha_s * (self.alpha_f / self.alpha_s) ** (ep / (self.epochs - 1))
            for x in X_norm:
                x = x.reshape(-1, 1)
                y = W @ x
                for i in range(self.num_components):
                    # Project out already learned components
                    proj = y[:i+1].T @ W[:i+1]
                    delta_w = lr * y[i] * (x.T - proj)
                    W[i:i+1] += delta_w

            print(f"Epoch {ep + 1}/{self.epochs}: learning rate alpha_t = {lr:.6f}")

        self.W = W  # Principal components matrix

    def encode(self, X, k=None):
        # Project input data onto the first k principal components
        if k is None:
            k = self.num_components
        X_norm = (X - self.mean) / self.std
        return self.W[:k] @ X_norm.T

    def decode(self, Y, k=None):
        # Reconstruct from compressed representation using top k components
        if k is None:
            k = self.num_components
        return (self.W[:k].T @ Y).T * self.std + self.mean

    def compress_and_reconstruct(self, X, k):
        # Perform both compression and reconstruction
        Y = self.encode(X, k)
        return self.decode(Y, k)

    def get_components(self):
        # Return learned principal components
        return self.W.copy()