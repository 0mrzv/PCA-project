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


# Model 1: Linear Autoencoder with separate weights

class LinearAutoencoderSeparate:
    def __init__(self, dim_inp, dim_hid, alpha_s=1e-2, alpha_f=1e-4, epochs=100, l2_reg=1e-4):
        self.dim_inp = dim_inp
        self.dim_hid = dim_hid
        self.alpha_s = alpha_s
        self.alpha_f = alpha_f
        self.epochs = epochs
        self.l2_reg = l2_reg
        self.loss_history = []
        self.X_mean = None
        self.X_std = None

        self.W_enc = np.random.randn(dim_hid, dim_inp) * 0.01
        self.W_enc -= np.mean(self.W_enc, axis=1, keepdims=True)

        self.W_dec = np.random.randn(dim_inp, dim_hid) * 0.01
        self.W_dec -= np.mean(self.W_dec, axis=1, keepdims=True)

    def fit(self, X):
        # Normalize input
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-5
        X = (X - self.X_mean) / self.X_std

        for ep in range(self.epochs):
            lr = self.alpha_s * (self.alpha_f / self.alpha_s) ** (ep / (self.epochs - 1))

            # Batch forward pass
            Y = self.W_enc @ X.T
            X_hat = self.W_dec @ Y
            error = X_hat - X.T
            loss = np.mean(np.sum(error ** 2, axis=0))
            self.loss_history.append(loss)

            # Batch gradients with L2 regularization
            dW_dec = (error @ Y.T) / X.shape[0] + self.l2_reg * self.W_dec
            dW_enc = (self.W_dec.T @ error @ X) / X.shape[0] + self.l2_reg * self.W_enc

            # Gradient updates
            self.W_dec -= lr * dW_dec
            self.W_enc -= lr * dW_enc

            # Normalize weights
            self.W_dec /= (np.linalg.norm(self.W_dec, axis=1, keepdims=True) + 1e-5)
            self.W_enc /= (np.linalg.norm(self.W_enc, axis=1, keepdims=True) + 1e-5)

    def reconstruct(self, X):
        X = (X - self.X_mean) / self.X_std
        return (self.W_dec @ (self.W_enc @ X.T)).T


# Model 2: Linear Autoencoder with shared weight matrix

class LinearAutoencoderShared:
    def __init__(self, dim_inp, dim_hid, alpha_s=1e-2, alpha_f=1e-4, epochs=100):
        self.dim_inp = dim_inp
        self.dim_hid = dim_hid
        self.alpha_s = alpha_s
        self.alpha_f = alpha_f
        self.epochs = epochs
        self.loss_history = []
        self.X_mean = None
        self.X_std = None

        self.W = np.random.randn(dim_hid, dim_inp) * 0.01
        self.W -= np.mean(self.W, axis=1, keepdims=True)

    def fit(self, X):
        self.X_mean = np.mean(X, axis=0)
        self.X_std = np.std(X, axis=0) + 1e-5
        X = (X - self.X_mean) / self.X_std

        for ep in range(self.epochs):
            lr = self.alpha_s * (self.alpha_f / self.alpha_s) ** (ep / (self.epochs - 1))
            Z = self.W @ X.T
            X_hat = self.W.T @ Z
            error = X_hat - X.T
            loss = np.mean(np.sum(error ** 2, axis=0))
            self.loss_history.append(loss)

            dW = 2 * (error @ X @ self.W.T).T / X.shape[0]
            self.W -= lr * dW

            self.W /= (np.linalg.norm(self.W, axis=1, keepdims=True) + 1e-5)
            print(f"Epoch {ep + 1}/{self.epochs}: learning rate alpha_t = {lr:.6f}")

    def reconstruct(self, X):
        X = (X - self.X_mean) / self.X_std
        return (self.W.T @ (self.W @ X.T)).T