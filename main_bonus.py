from linear_autoenc import *
import matplotlib
matplotlib.use("TkAgg")


# Load image and normalize
block_size = 8
X_blocks, img_shape, original_img = load_image_blocks("elaine.256.png", block_size)
X = X_blocks.astype(np.float32)

# Train both models
model_sep = LinearAutoencoderSeparate(64, 32, alpha_s=1e-2, alpha_f=1e-4, epochs=200)
model_shr = LinearAutoencoderShared(64, 32, alpha_s=1e-2, alpha_f=1e-4, epochs=200)

model_sep.fit(X)
model_shr.fit(X)

# Reconstruct
recon_sep = model_sep.reconstruct(X) * model_sep.X_std + model_sep.X_mean
recon_shr = model_shr.reconstruct(X) * model_shr.X_std + model_shr.X_mean

mse_sep = compute_mse(X_blocks, recon_sep)
mse_shr = compute_mse(X_blocks, recon_shr)

img_sep = rebuild_image_from_blocks(recon_sep, img_shape)
img_shr = rebuild_image_from_blocks(recon_shr, img_shape)


# Plots of MSE per epoch and reconstruction results
plt.figure(figsize=(10, 5))
plt.plot(model_sep.loss_history, label="Separate Weights AE")
plt.plot(model_shr.loss_history, label="Shared Weights AE")
plt.title("Training Loss (MSE) over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.yscale("log")
plt.legend()
plt.tight_layout()
plt.show()


fig, axes = plt.subplots(1, 3, figsize=(15, 5))
titles = ["Original Image", f"Separate Weights AE\nMSE: {mse_sep:.2f}", f"Shared Weights AE\nMSE: {mse_shr:.2f}"]
images = [original_img, img_sep, img_shr]

for ax, title, img in zip(axes, titles, images):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.suptitle("Comparison of Linear Autoencoders")
plt.tight_layout()
plt.show()

# Test the model on two new images
new_img = ["lena.tif", "woman2.tif"]
titles = ["Lena", "Woman2"]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))

for idx, path in enumerate(new_img):
    blocks_new, shape_new, orig_new = load_image_blocks(path, block_size)
    X_new = blocks_new.astype(np.float32)

    recon_sep = model_sep.reconstruct(X_new) * model_sep.X_std + model_sep.X_mean
    recon_shr = model_shr.reconstruct(X_new) * model_shr.X_std + model_shr.X_mean

    mse_sep_new = compute_mse(blocks_new, recon_sep)
    mse_shr_new = compute_mse(blocks_new, recon_shr)

    img_sep_new = rebuild_image_from_blocks(recon_sep, shape_new)
    img_shr_new = rebuild_image_from_blocks(recon_shr, shape_new)

    images = [orig_new, img_sep_new, img_shr_new]
    labels = ["Original", f"Separate AE\nMSE: {mse_sep_new:.2f}", f"Shared AE\nMSE: {mse_shr_new:.2f}"]

    for col, (img, label) in enumerate(zip(images, labels)):
        axes[idx, col].imshow(img, cmap='gray')
        axes[idx, col].set_title(label)
        axes[idx, col].axis('off')

plt.suptitle("Testing: Separate vs Shared Linear Autoencoders")
plt.tight_layout()
plt.show()