import os
import matplotlib.pyplot as plt

# Render images in batches of up to 25
def render_images_in_batches(images, generation_confidences, generation_ssims_for_target, generation_ssims_for_similarity, generation_nccs_for_target, generation_nccs_for_similarity, indices, output_subdir, target_class_for_confidence, replicate, experiment_number, model_names):
    max_images_per_grid = 25
    total_images = len(images)
    num_batches = (total_images + max_images_per_grid - 1) // max_images_per_grid

    for batch_idx in range(num_batches):
        start_idx = batch_idx * max_images_per_grid
        end_idx = min(start_idx + max_images_per_grid, total_images)
        
        fig, axs = plt.subplots(5, 5, figsize=(20, 20))
        fig.subplots_adjust(hspace=0.7, wspace=0.4)

        for i, (img, confs, ssimT, ssimS, nccT, nccS, idx) in enumerate(zip(images[start_idx:end_idx], generation_confidences[start_idx:end_idx], generation_ssims_for_target[start_idx:end_idx], generation_ssims_for_similarity[start_idx:end_idx], generation_nccs_for_target[start_idx:end_idx], generation_nccs_for_similarity[start_idx:end_idx], indices[start_idx:end_idx])):
            row, col = divmod(i, 5)
            axs[row, col].imshow(img, cmap='gray')
            title_lines = [f'Gen {idx}']
            for model_name, conf in zip(model_names, confs):
                title_lines.append(f'{model_name} Conf: {conf:.4f}')
            title_lines.append(f'SSIM-{target_class_for_confidence}: {ssimT:.4f}')
            title_lines.append(f'NCC-{target_class_for_confidence}: {nccT:.4f}')
            axs[row, col].set_title("\n".join(title_lines), fontsize=8)
            axs[row, col].axis('off')

        for j in range(len(images[start_idx:end_idx]), 25):
            row, col = divmod(j, 5)
            axs[row, col].axis('off')

        plt.savefig(os.path.join(output_subdir, f"class_{target_class_for_confidence}_rep{replicate}_batch_{batch_idx + 1}.png"))
        plt.close(fig)

# Plot scores over generations
def plot_scores_vs_generations(generations, generation_confidences, generation_ssims_for_target, generation_ssims_for_similarity, generation_nccs_for_target, generation_nccs_for_similarity, output_subdir, target_class_for_confidence, replicate, model_names):
    plt.figure(figsize=(10, 6))
    for model_name, conf_scores in zip(model_names, zip(*generation_confidences)):
        plt.plot(generations, conf_scores, label=f'{model_name} Confidence', marker='o')

    plt.plot(generations, generation_ssims_for_target, label=f'SSIM (Target)', marker='s')
    plt.plot(generations, generation_nccs_for_target, label=f'NCC (Target)', marker='^')

    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title(f"Scores vs. Generations (Target Class: {target_class_for_confidence}, Replicate: {replicate})")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_subdir, f"class_{target_class_for_confidence}_rep{replicate}_scores_plot.png"))
    plt.close()
