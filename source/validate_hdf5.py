import os
import h5py
import matplotlib.pyplot as plt
import numpy as np

# Paths for HDF5 file and output directories
file_path = "/home/ankit-gupta/Work/Projects/Active/projectEvoFool/data_generated/Exp_1/sklearnDigits/GBM/class_0/replicate_2/evolved_images.hdf5"
output_best_dir = "output_best_images"
output_population_dir = "output_final_population"

# Create output directories
os.makedirs(output_best_dir, exist_ok=True)
os.makedirs(output_population_dir, exist_ok=True)

# Function to save images in grids with detailed tags
def save_images_in_grids_with_tags(images, params, output_dir, batch_name, grid_size=25, img_shape=(8, 8), title_prefix=""):
    """
    Save images in grids of up to grid_size images per batch with tags.
    Args:
        images (ndarray): Array of images.
        params (list of str): List of strings to tag each image.
        output_dir (str): Directory to save the grids.
        batch_name (str): Name of the batch for file naming.
        grid_size (int): Maximum number of images per grid.
        img_shape (tuple): Shape of each image (height, width).
        title_prefix (str): Prefix for the title of each image.
    """
    num_images = images.shape[0]
    num_batches = (num_images + grid_size - 1) // grid_size  # Ceiling division

    for batch_idx in range(num_batches):
        start_idx = batch_idx * grid_size
        end_idx = min(start_idx + grid_size, num_images)
        batch_images = images[start_idx:end_idx]
        batch_params = params[start_idx:end_idx]

        # Create a grid for the batch
        rows = cols = int(np.ceil(np.sqrt(len(batch_images))))  # Square grid dimensions
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        fig.subplots_adjust(hspace=0.6, wspace=0.4)

        for i, ax in enumerate(axes.flatten()):
            if i < len(batch_images):
                image = batch_images[i].reshape(img_shape)  # Reshape image if needed
                ax.imshow(image, cmap="gray")
                ax.axis("off")
                ax.set_title(f"{title_prefix} {batch_params[i]}", fontsize=8)
            else:
                ax.axis("off")  # Turn off unused subplots

        # Save the grid
        save_path = os.path.join(output_dir, f"{batch_name}_batch_{batch_idx + 1}.png")
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
        plt.close(fig)

# Open the HDF5 file
with h5py.File(file_path, "r") as hdf5_file:
    # Validate structure
    print("Validating HDF5 Structure...")
    datasets = list(hdf5_file.keys())
    attributes = dict(hdf5_file.attrs)
    
    print("\nDatasets in HDF5 file:")
    print(datasets)
    
    print("\nAttributes in HDF5 file:")
    print(attributes)
    
    # Validate intermediate data
    print("\nValidating Intermediate Data...")
    images = hdf5_file["images"][:]
    gen_counts = hdf5_file["genCount"][:]
    fitness = hdf5_file["fitness"][:]
    confidence_target = hdf5_file["confidence_target"][:]
    confidence_similarity = hdf5_file["confidence_similarity"][:]
    ssim_target = hdf5_file["ssim_target"][:]
    ssim_similarity = hdf5_file["ssim_similarity"][:]
    ncc_target = hdf5_file["ncc_target"][:]
    ncc_similarity = hdf5_file["ncc_similarity"][:]
    
    # Prepare tags for intermediate images
    best_image_tags = [
        f"Gen: {gen}, Fit: {fit:.2f}, \nConf_T: {conf_t:.2f}, SSIM_T: {ssim_t:.2f}"
        for gen, fit, conf_t, ssim_t in zip(gen_counts, fitness, confidence_target, ssim_target)
    ]
    
    # Save intermediate best images as grids with tags
    print("\nSaving Intermediate Best Images as Grids with Tags...")
    save_images_in_grids_with_tags(images, best_image_tags, output_best_dir, batch_name="best_images", grid_size=25, img_shape=(8, 8))

    # Validate and save final population
    if "final_population" in hdf5_file:
        final_population = hdf5_file["final_population"][:]
        print("\nFinal Population Data:")
        print(f"Final Population Shape: {final_population.shape}")
        final_genCount = hdf5_file.attrs["final_genCount"]
        print(f"Final Generation Count: {final_genCount}")
        
        # Prepare tags for final population
        final_population_tags = [f"Ind: {i}" for i in range(final_population.shape[0])]
        
        # Save final population images as grids with tags
        print("\nSaving Final Population Images as Grids...")
        save_images_in_grids_with_tags(final_population, final_population_tags, output_population_dir, batch_name="final_population", grid_size=25, img_shape=(8, 8))
    else:
        print("\nFinal Population Data: Not Found")

print("\nValidation and Saving Completed.")
