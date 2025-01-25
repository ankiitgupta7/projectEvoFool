import os
import csv
import numpy as np
import h5py
import tensorflow as tf
from metrics import compute_ncc, compute_ssim
from visualisation import render_images_in_batches, plot_scores_vs_generations
from deap import tools  # Add this import
import json
import time
import random
from tqdm import tqdm

def save_run_summary(output_subdir, early_stop_gen, experiment_details, tqdm_details):
    """
    Save a run summary file with optimized details of the completed run.
    """
    summary_path = os.path.join(output_subdir, "run_summary.json")
    summary = {
        "early_stop_generation": early_stop_gen,
        "experiment_details": experiment_details,
        "progress_details": {
            "current_generation": tqdm_details["current"],
            "total_generations": tqdm_details["total"],
            "rate": tqdm_details["rate"],
            "elapsed_time_seconds": tqdm_details["elapsed_time"],
        }
    }
    with open(summary_path, "w") as summary_file:
        json.dump(summary, summary_file, indent=4)

def collect_scores(individual, models, input_shape, target_image, image_for_similarity_comarison):
    """
    Compute confidence scores for all models and SSIM/NCC for a given individual.
    """
    image = np.array(individual).reshape(*input_shape)

    if not (0 <= image.min() <= image.max() <= 1):
        raise ValueError("Image values are not normalized.")

    confidences = {}
    for model_name, model in models.items():
        if isinstance(model, tf.keras.Model):  # TensorFlow model check
            image_expanded = image[np.newaxis, ..., np.newaxis]
            probabilities = model.predict(image_expanded, verbose=0)
        else:  # Sklearn or similar model check
            image_flat = image.flatten().reshape(1, -1)
            probabilities = model.predict_proba(image_flat)
        confidences[model_name] = probabilities[0]

    # Compute SSIM
    ssim_score_similarity_class = compute_ssim(image, image_for_similarity_comarison)
    ssim_score_target_class = compute_ssim(image, target_image)

    # Compute NCC
    ncc_score_similarity_class = compute_ncc(image, image_for_similarity_comarison)
    ncc_score_target_class = compute_ncc(image, target_image)

    return confidences, ssim_score_target_class, ssim_score_similarity_class, ncc_score_target_class, ncc_score_similarity_class

def run_evolution(experiment_number, toolbox, ngen, trained_models, input_shape, target_class_for_confidence, target_class_for_similarity, similarity_metric, output_subdir, generation_interval, replicate, target_image, image_for_similarity_comarison, model_names, dataset_name, seed):
    """
    Run the evolutionary algorithm to optimize images.
    """
    start_time = time.time()
    early_stop_gen = None

    # Set random seeds
    np.random.seed(seed % (2**32))
    tf.random.set_seed(seed)
    random.seed(seed)

    # Create output directories
    os.makedirs(output_subdir, exist_ok=True)
    batch_dir = os.path.join(output_subdir, "batch_image_grid")
    os.makedirs(batch_dir, exist_ok=True)

    # Initialize HDF5 file
    hdf5_path = os.path.join(output_subdir, "evolved_images.hdf5")
    with h5py.File(hdf5_path, "w") as hdf5_file:
        hdf5_file.create_dataset("images", shape=(0, *input_shape), maxshape=(None, *input_shape), dtype="float32", compression="gzip")
        hdf5_file.create_dataset("genCount", shape=(0,), maxshape=(None,), dtype="int32", compression="gzip")
        hdf5_file.create_dataset("fitness", shape=(0,), maxshape=(None,), dtype="float32", compression="gzip")

    population = toolbox.population(n=50)

    # Initialize lists to store metrics across generations
    generation_confidences = []
    generation_ssims_for_target = []
    generation_ssims_for_similarity = []
    generation_nccs_for_target = []
    generation_nccs_for_similarity = []

    # CSV for logging scores
    scores_csv_path = os.path.join(output_subdir, f'scores_class_{target_class_for_confidence}_rep{replicate}.csv')
    with open(scores_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        header = ["Generation"] + [f"Confidence_{model}" for model in model_names] + ["SSIM_Target", "SSIM_Similarity", "NCC_Target", "NCC_Similarity"]
        writer.writerow(header)

        progress_bar = tqdm(range(ngen + 1), desc="Evolving Generations", unit="gen")
        try:
            for gen in progress_bar:
                # Selection
                offspring = toolbox.select(population, len(population))
                offspring = list(map(toolbox.clone, offspring))

                # Mutation
                for mutant in offspring:
                    if random.random() < 0.2:
                        toolbox.mutate(mutant)
                        del mutant.fitness.values

                # Repair
                for ind in offspring:
                    for i in range(len(ind)):
                        ind[i] = np.clip(ind[i], 0, 1)

                # Evaluation
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                for ind in invalid_ind:
                    confidences, ssim_target, ssim_similarity, ncc_target, ncc_similarity = collect_scores(
                        ind, trained_models, input_shape, target_image, image_for_similarity_comarison
                    )
                    ind.fitness.values = (ssim_target,)  # Use SSIM as fitness

                # Replace population
                population[:] = offspring

                # Log best individual of generation
                best_ind = tools.selBest(population, 1)[0]
                confidences, ssim_target, ssim_similarity, ncc_target, ncc_similarity = collect_scores(
                    best_ind, trained_models, input_shape, target_image, image_for_similarity_comarison
                )

                # Store metrics for the generation
                generation_confidences.append([confidences[model][target_class_for_confidence] for model in model_names])
                generation_ssims_for_target.append(ssim_target)
                generation_ssims_for_similarity.append(ssim_similarity)
                generation_nccs_for_target.append(ncc_target)
                generation_nccs_for_similarity.append(ncc_similarity)

                # Save scores
                row = [gen] + generation_confidences[-1] + [ssim_target, ssim_similarity, ncc_target, ncc_similarity]
                writer.writerow(row)
                file.flush()

                # Save images and fitness to HDF5
                with h5py.File(hdf5_path, "a") as hdf5_file:
                    hdf5_file["images"].resize((hdf5_file["images"].shape[0] + 1), axis=0)
                    hdf5_file["images"][-1] = np.array(best_ind).reshape(*input_shape)
                    hdf5_file["genCount"].resize((hdf5_file["genCount"].shape[0] + 1), axis=0)
                    hdf5_file["genCount"][-1] = gen
                    hdf5_file["fitness"].resize((hdf5_file["fitness"].shape[0] + 1), axis=0)
                    hdf5_file["fitness"][-1] = ssim_target

                # Render image grid and plot
                if gen % generation_interval == 0 or gen == ngen:
                    render_images_in_batches(
                        images=[np.array(ind).reshape(*input_shape) for ind in population],
                        generation_confidences=generation_confidences,
                        generation_ssims_for_target=generation_ssims_for_target,
                        generation_ssims_for_similarity=generation_ssims_for_similarity,
                        generation_nccs_for_target=generation_nccs_for_target,
                        generation_nccs_for_similarity=generation_nccs_for_similarity,
                        indices=list(range(len(population))),
                        output_subdir=batch_dir,
                        target_class_for_confidence=target_class_for_confidence,
                        replicate=replicate,
                        experiment_number=experiment_number,
                        model_names=model_names
                    )

                    plot_scores_vs_generations(
                        list(range(gen + 1)),
                        generation_confidences,
                        generation_ssims_for_target,
                        generation_ssims_for_similarity,
                        generation_nccs_for_target,
                        generation_nccs_for_similarity,
                        output_subdir,
                        target_class_for_confidence,
                        replicate,
                        model_names
                    )
                
                # Optional: Terminate if fitness reaches a threshold
                if ssim_target >= 0.99:
                    tqdm.write(f"Stopping early at generation {gen} as fitness threshold reached.")
                    early_stop_gen = gen

                    break

        finally:
            progress_bar.close()

    total_time = time.time() - start_time
    experiment_details = {
        "experiment_number": experiment_number,
        "dataset_name": dataset_name,
        "target_class_for_confidence": target_class_for_confidence,
        "target_class_for_similarity": target_class_for_similarity,
        "similarity_metric": similarity_metric,
        "replicate": replicate,
        "seed": seed,
    }

    save_run_summary(output_subdir, early_stop_gen, experiment_details, {
        "current": ngen,
        "total": ngen,
        "rate": ngen / total_time,
        "elapsed_time": total_time
    })
    print(f"Run completed in {total_time:.2f} seconds. Results saved to: {output_subdir}")
