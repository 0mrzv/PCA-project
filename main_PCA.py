from GHA_PCA import *
import matplotlib
matplotlib.use("TkAgg")

block_size = 8
blocks, image_shape, original_image = load_image_blocks("elaine.256.png", block_size)

# Initialize and train the model using 16 components
model = GHA_PCA(num_components=16, alpha_s=1e-3, alpha_f=1e-5, epochs=200)
model.fit(blocks)

# Normalize the training blocks
X = blocks.astype(np.float32)
X_mean = np.mean(X, axis=0)
X_std = np.std(X, axis=0) + 1e-5
X_norm = (X - X_mean) / X_std

# PCA correlation matrix
correlation_matrix = (X_norm.T @ X_norm) / X_norm.shape[0]  # shape: (64, 64)

# Eigenvalues in decreasing order
eigenvalues, _ = np.linalg.eigh(correlation_matrix)  # for symmetric matrices
eigenvalues = sorted(eigenvalues, reverse=True)

plt.figure()
plt.plot(np.arange(1, len(eigenvalues) + 1), eigenvalues, marker='o')
plt.title("Eigenvalues of the PCA Correlation Matrix")
plt.xlabel("Principal Component Index")
plt.ylabel("Eigenvalue")
plt.tight_layout()
plt.show()


# Top 8 learned principal components as 8x8 images
top_components = model.get_components()[:8].reshape((8, 8, 8))

fig, axes = plt.subplots(1, 8, figsize=(16, 3))
for i, ax in enumerate(axes):
    ax.imshow(top_components[i], cmap='gray')
    ax.set_title(f"PC {i+1}")
    ax.axis('off')
plt.suptitle("Top 8 Principal Components")
plt.show()


# Cumulative variance explained
eigenvalues, _ = np.linalg.eigh(correlation_matrix)
eigenvalues = sorted(eigenvalues, reverse=True)
cumulative = np.cumsum(eigenvalues) / np.sum(eigenvalues)

optimal_k = np.argmax(cumulative >= 0.95) + 1  # +1 because indices start at 0

print(f"Minimum number of components to retain {95:.0f}% variance: k = {optimal_k}")

plt.figure()
plt.plot(np.arange(1, len(cumulative)+1), cumulative, marker='o')
plt.axhline(0.95, linestyle='--', color='r', label='95% threshold')
plt.axvline(optimal_k, linestyle='--', color='g', label=f'k = {optimal_k}')
plt.title("Cumulative Variance Explained")
plt.xlabel("Number of Principal Components")
plt.ylabel("Cumulative Variance Ratio")
plt.legend()
plt.tight_layout()
plt.show()


# Reconstruction for different k values
k_values = [8, 16, 32]
reconstructed_versions = {}
for k in k_values:
    recon_blocks = model.compress_and_reconstruct(blocks, k)
    recon_img = rebuild_image_from_blocks(recon_blocks, image_shape, block_size)
    reconstructed_versions[k] = recon_img

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')
for i, k in enumerate(k_values):
    axes[i + 1].imshow(reconstructed_versions[k], cmap='gray')
    axes[i + 1].set_title(f"k = {k}")
    axes[i + 1].axis('off')
plt.suptitle("Reconstruction for k = 8, 16, 32")
plt.tight_layout()
plt.show()

# Progressive reconstruction
k_values = [1] + list(range(2, 17, 2))
reconstructed_versions = {}
mse_progressive = []
for k in k_values:
    recon_blocks = model.compress_and_reconstruct(blocks, k)
    recon_img = rebuild_image_from_blocks(recon_blocks, image_shape, block_size)
    reconstructed_versions[k] = recon_img
    mse = compute_mse(original_image, recon_img)
    mse_progressive.append(mse)

rows, cols = 2, 5
fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))  # Bigger images
axes = axes.flatten()
axes[0].imshow(original_image, cmap='gray')
axes[0].set_title("Original")
axes[0].axis('off')
for i, k in enumerate(k_values):
    # ax = axes[i // cols, i % cols]
    ax = axes[i + 1]
    ax.imshow(reconstructed_versions[k], cmap='gray')
    ax.set_title(f"k = {k}")
    ax.axis('off')

plt.suptitle("Progressive Reconstruction for k = 1 to 16")
plt.tight_layout()
plt.show()

mse_16 = mse_progressive[k_values.index(16)]

# Plot of MSE vs. number of components
plt.figure()
plt.plot(k_values, mse_progressive, marker='o')
plt.title("MSE vs Number of Components")
plt.xlabel("Number of Components (k)")
plt.ylabel("Mean Squared Error")
plt.axhline(mse_16, linestyle='--', color='r', label=f'MSE = {mse_16:.2f}')
plt.legend()
plt.tight_layout()
plt.show()


# Coefficient activation maps for top 8 PCs

encoded = model.encode(blocks, k=8)  # (k, n_blocks)
encoded_maps = encoded.reshape(8, 32, 32)  # k=8, blocks in 32x32 grid (since 256/8 = 32)

fig, axes = plt.subplots(1, 8, figsize=(16, 2))
for i in range(8):
    axes[i].imshow(encoded_maps[i], cmap='gray')
    axes[i].set_title(f"PC {i+1}")
    axes[i].axis('off')
plt.suptitle("32×32 Coefficient Maps for Top 8 Components")
plt.tight_layout()
plt.show()


# Testing the model on new portraits

blocks_lena, shape_lena, lena_orig  = load_image_blocks("lena.tif")
blocks_woman2, shape_woman2, woman2_orig = load_image_blocks("woman2.tif")

# We use previously trained PCA components to encode & reconstruct the new images
k = 16
recon_lena = model.compress_and_reconstruct(blocks_lena, k)
recon_woman2 = model.compress_and_reconstruct(blocks_woman2, k)

lena_reconstructed = rebuild_image_from_blocks(recon_lena, shape_lena, block_size)
woman2_reconstructed = rebuild_image_from_blocks(recon_woman2, shape_woman2, block_size)

mse_lena = compute_mse(lena_orig, lena_reconstructed)
mse_woman2 = compute_mse(woman2_orig, woman2_reconstructed)

test_images = [("Lena", lena_orig, lena_reconstructed, mse_lena),
               ("Woman2", woman2_orig, woman2_reconstructed, mse_woman2)]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))

for row, (label, original, reconstructed, mse) in enumerate(test_images):
    axes[row, 0].imshow(original, cmap='gray')
    axes[row, 0].set_title(f"Original {label}")
    axes[row, 0].axis('off')

    axes[row, 1].imshow(reconstructed, cmap='gray')
    axes[row, 1].set_title(f"Reconstructed {label} (k={k})\nMSE: {mse:.2f}")
    axes[row, 1].axis('off')

plt.suptitle("Testing: PCA using Generalized Hebbian Algorithm")
plt.tight_layout()
plt.show()

# Train separate GHA PCA on Lena
model_lena = GHA_PCA(num_components=k, alpha_s=1e-2, alpha_f=1e-4, epochs=200)
model_lena.fit(blocks_lena)
W_elaine = model.get_components()
W_lena = model_lena.get_components()

# Cosine similarity between learned components
similarity_matrix = np.abs(W_elaine @ W_lena.T)

plt.figure()
plt.imshow(similarity_matrix, cmap='viridis')
plt.title("Component Similarity (Elaine vs Lena)")
plt.colorbar(label="|cosine similarity|")
plt.xlabel("Lena Components")
plt.ylabel("Elaine Components")
plt.tight_layout()
plt.show()