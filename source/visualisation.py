import os
import matplotlib.pyplot as plt

# Render images in batches of up to 25
def render_images_in_batches(images, generation_confidences_for_target, generation_confidences_for_similarity, generation_ssims_for_target, generation_ssims_for_similarity, generation_nccs_for_target, generation_nccs_for_similarity, indices, output_subdir, target_digit_for_confidence, target_digit_for_similarity, replicate):
    max_images_per_grid = 25
    total_images = len(images)
    num_batches = (total_images + max_images_per_grid - 1) // max_images_per_grid

    for batch_idx in range(num_batches):
        start_idx = batch_idx * max_images_per_grid
        end_idx = min(start_idx + max_images_per_grid, total_images)
        
        fig, axs = plt.subplots(5, 5, figsize=(20, 20))
        fig.subplots_adjust(hspace=0.7, wspace=0.4)

        for i, (img, confT, confS, ssimT, ssimS, nccT, nccS, idx) in enumerate(zip(images[start_idx:end_idx], generation_confidences_for_target[start_idx:end_idx], generation_confidences_for_similarity[start_idx:end_idx], generation_ssims_for_target[start_idx:end_idx], generation_ssims_for_similarity[start_idx:end_idx], generation_nccs_for_target[start_idx:end_idx], generation_nccs_for_similarity[start_idx:end_idx], indices[start_idx:end_idx])):
            row, col = divmod(i, 5)
            axs[row, col].imshow(img, cmap='gray')
            axs[row, col].set_title(f'Gen {idx}\nConf-{target_digit_for_confidence}: {confT:.4f}, Conf-{target_digit_for_similarity}: {confS:.4f}\nSSIM-{target_digit_for_confidence}: {ssimT:.4f}, SSIM-{target_digit_for_similarity}: {ssimS:.4f} \nNCC-{target_digit_for_confidence}: {nccT:.4f}, NCC-{target_digit_for_similarity}: {nccS:.4f}', fontsize=8)
            axs[row, col].axis('off')

        for j in range(len(images[start_idx:end_idx]), 25):
            row, col = divmod(j, 5)
            axs[row, col].axis('off')

        plt.savefig(os.path.join(output_subdir, f"digit_{target_digit_for_confidence}_rep{replicate}_batch_{batch_idx + 1}.png"))
        plt.close(fig)

# Plot scores over generations
def plot_scores_vs_generations(generations, generation_confidences_for_target, generation_confidences_for_similarity, generation_ssims_for_target, generation_ssims_for_similarity, generation_nccs_for_target, generation_nccs_for_similarity, output_subdir, target_digit_for_confidence, target_digit_for_similarity, replicate, model_name, dataset_name):
    plt.figure(figsize=(10, 6))
    plt.plot(generations, generation_confidences_for_target, label=f'Confidence (Target Class: {target_digit_for_confidence})', marker='o')
    plt.plot(generations, generation_confidences_for_similarity, label=f'Confidence (Similarity Class: {target_digit_for_similarity})', marker='o')
    plt.plot(generations, generation_ssims_for_target, label=f'SSIM (Target Class: {target_digit_for_confidence})', marker='s')
    plt.plot(generations, generation_ssims_for_similarity, label=f'SSIM (Similarity Class: {target_digit_for_similarity})', marker='s')
    plt.plot(generations, generation_nccs_for_target, label=f'NCC (Target Class: {target_digit_for_confidence})', marker='^')
    plt.plot(generations, generation_nccs_for_similarity, label=f'NCC (Similarity Class: {target_digit_for_similarity})', marker='^')

    plt.xlabel("Generation")
    plt.ylabel("Score")
    plt.title(f"Scores vs. Gen. (Model/Dataset: {model_name}/{dataset_name}, Target Class: {target_digit_for_confidence}, Similarity Class: {target_digit_for_similarity}, Replicate: {replicate})")
    plt.legend()
    plt.grid(True)

    plt.savefig(os.path.join(output_subdir, f"digit_{target_digit_for_confidence}_rep{replicate}_scores_plot.png"))
    plt.close()
