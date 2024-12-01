import os
import matplotlib.pyplot as plt

# Render images in batches of up to 25
def render_images_in_batches(images, confidences, ssims, nccs, indices, output_subdir, target_digit, replicate):
    max_images_per_grid = 25
    total_images = len(images)
    num_batches = (total_images + max_images_per_grid - 1) // max_images_per_grid

    for batch_idx in range(num_batches):
        start_idx = batch_idx * max_images_per_grid
        end_idx = min(start_idx + max_images_per_grid, total_images)
        
        fig, axs = plt.subplots(5, 5, figsize=(20, 20))
        fig.subplots_adjust(hspace=0.7, wspace=0.4)

        for i, (img, conf, ssim, ncc, idx) in enumerate(zip(images[start_idx:end_idx], confidences[start_idx:end_idx], ssims[start_idx:end_idx], nccs[start_idx:end_idx], indices[start_idx:end_idx])):
            row, col = divmod(i, 5)
            axs[row, col].imshow(img, cmap='gray')
            axs[row, col].set_title(f'Gen {idx}\nConf: {conf:.3f}\nSSIM: {ssim:.3f} \nNCC: {ncc:.3f}', fontsize=8)
            axs[row, col].axis('off')

        for j in range(len(images[start_idx:end_idx]), 25):
            row, col = divmod(j, 5)
            axs[row, col].axis('off')

        plt.savefig(os.path.join(output_subdir, f"digit_{target_digit}_rep{replicate}_batch_{batch_idx + 1}.png"))
        plt.close(fig)

# Plot scores over generations
def plot_scores_vs_generations(generations, confidences, ssims, nccs, output_subdir, target_digit, replicate, model_name):
    plt.figure(figsize=(10, 6))
    plt.plot(generations, confidences, label='Confidence', marker='o')
    plt.plot(generations, ssims, label='SSIM', marker='s')
    plt.plot(generations, nccs, label='NCC', marker='^')

    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title(f"Scores vs. Generations (Model: {model_name}, Digit: {target_digit}, Replicate: {replicate})")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_subdir, f"digit_{target_digit}_rep{replicate}_scores_plot.png"))
    plt.close()
