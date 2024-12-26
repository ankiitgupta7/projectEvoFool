import os
import csv
import numpy as np
import h5py
import tensorflow as tf  # Import TensorFlow
from deap import tools  # Import tools from DEAP
from metrics import compute_ncc, compute_ssim
from visualisation import render_images_in_batches, plot_scores_vs_generations
import json
import time
from tqdm import tqdm

def save_run_summary(output_subdir, early_stop_gen, experiment_details, tqdm_details):
    """
    Save a run summary file with optimized details of the completed run.
    Args:
        output_subdir (str): Path to the output directory.
        ngen (int): Total generations.
        early_stop_gen (int): Generation at which the run stopped early (or None if completed).
        total_time (float): Total runtime in seconds.
        experiment_details (dict): Additional details of the experiment.
        tqdm_details (dict): Optimized tqdm progress details.
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


# Collect scores (confidence, SSIM, NCC)
def collect_scores(individual, model, input_shape, target_image, image_for_similarity_comarison, target_digit_for_confidence, target_digit_for_similarity):
    """
    Compute confidence, SSIM, and NCC scores for a given individual.
    """
    image = np.array(individual).reshape(*input_shape)

    if not (0 <= image.min() <= image.max() <= 1):
        raise ValueError("Image values are not normalized.")
    if not (0 <= image_for_similarity_comarison.min() <= image_for_similarity_comarison.max() <= 1):
        raise ValueError("Median image values are not normalized.")

    # Compute confidence
    if isinstance(model, tf.keras.Model):  # TensorFlow model check
        image_expanded = np.array(individual).reshape(1, *input_shape, 1)
        probabilities = model.predict(image_expanded, verbose=0)
    else:  # Sklearn or similar model check
        image_expanded = np.array(individual).reshape(1, -1)
        probabilities = model.predict_proba(image_expanded)

    confidence_score_for_target_digit = probabilities[0][target_digit_for_confidence]

    confidence_score_for_similarity_digit = probabilities[0][target_digit_for_similarity]

    confidence_scores = probabilities[0]

    # Compute SSIM
    ssim_score_similarity_class = compute_ssim(image, image_for_similarity_comarison)
    ssim_score_target_class = compute_ssim(image, target_image)

    # Compute NCC
    ncc_score_similarity_class = compute_ncc(image, image_for_similarity_comarison)
    ncc_score_target_class = compute_ncc(image, target_image)

    return confidence_score_for_target_digit, confidence_score_for_similarity_digit, ssim_score_target_class, ssim_score_similarity_class, ncc_score_target_class, ncc_score_similarity_class, confidence_scores

def initialize_hdf5_file(file_path, image_shape):
    """
    Initialize an HDF5 file for saving evolved image states.
    Args:
        file_path (str): Path to the HDF5 file.
        image_shape (tuple): Shape of the image array.
    """
    with h5py.File(file_path, "w") as hdf5_file:
        # Create datasets for images, metrics, and generation counts
        hdf5_file.create_dataset("images", shape=(0, *image_shape), maxshape=(None, *image_shape), dtype="float32", compression="gzip")
        hdf5_file.create_dataset("genCount", shape=(0,), maxshape=(None,), dtype="int32", compression="gzip")  # Store generation counts
        hdf5_file.create_dataset("fitness", shape=(0,), maxshape=(None,), dtype="float32", compression="gzip")
        hdf5_file.create_dataset("confidence_target", shape=(0,), maxshape=(None,), dtype="float32", compression="gzip")
        hdf5_file.create_dataset("confidence_similarity", shape=(0,), maxshape=(None,), dtype="float32", compression="gzip")
        hdf5_file.create_dataset("ssim_target", shape=(0,), maxshape=(None,), dtype="float32", compression="gzip")
        hdf5_file.create_dataset("ssim_similarity", shape=(0,), maxshape=(None,), dtype="float32", compression="gzip")
        hdf5_file.create_dataset("ncc_target", shape=(0,), maxshape=(None,), dtype="float32", compression="gzip")
        hdf5_file.create_dataset("ncc_similarity", shape=(0,), maxshape=(None,), dtype="float32", compression="gzip")

def append_to_hdf5(file_path, gen, image, fitness, confidence_target, confidence_similarity, ssim_target, ssim_similarity, ncc_target, ncc_similarity):
    """
    Append evolved image state data to an HDF5 file.
    Args:
        file_path (str): Path to the HDF5 file.
        gen (int): Generation count.
        image (ndarray): Image array.
        fitness (float): Fitness score.
        confidence_target (float): Confidence for the target class.
        confidence_similarity (float): Confidence for the similarity class.
        ssim_target (float): SSIM score for the target class.
        ssim_similarity (float): SSIM score for the similarity class.
        ncc_target (float): NCC score for the target class.
        ncc_similarity (float): NCC score for the similarity class.
    """
    with h5py.File(file_path, "a") as hdf5_file:
        # Append data to each dataset
        for key, value in {
            "genCount": [gen],  # Generation count
            "images": image[np.newaxis, ...],  # Add a new axis for batch dimension
            "fitness": [fitness],
            "confidence_target": [confidence_target],
            "confidence_similarity": [confidence_similarity],
            "ssim_target": [ssim_target],
            "ssim_similarity": [ssim_similarity],
            "ncc_target": [ncc_target],
            "ncc_similarity": [ncc_similarity]
        }.items():
            dataset = hdf5_file[key]
            dataset.resize(dataset.shape[0] + len(value), axis=0)
            dataset[-len(value):] = value

# Compute fitness based on scores
def fitness(experiment_number, confidence_score_for_target_digit, ssim_score_similarity_class, confidence_score_for_similarity_digit):
    """
    Compute the fitness of an individual based on the experiment number.
    """
    if experiment_number == "1": # Confidence as fitness - to see if the SSIM increases with increasing confidence for same class
        return confidence_score_for_target_digit
    elif experiment_number == "2_1a": # perceptible fooling 1: looks like target class but model has low confidence
        return ssim_score_similarity_class - confidence_score_for_target_digit
    # elif experiment_number == "2_1_2":  # perceptible fooling 2: looks like similarity class but model has high confidence for target class
    #     return (ssim_score_similarity_class + confidence_score_for_target_digit)/2
    elif experiment_number == "2_1b":  # perceptible fooling 2: looks like similarity class but model has high confidence for target class
        return (ssim_score_similarity_class + confidence_score_for_target_digit)/2 - (confidence_score_for_similarity_digit)/2
    elif experiment_number == "2_2":  # imperceptible fooling: does not look like target class (giberish) but model has high confidence
        return confidence_score_for_target_digit - abs(ssim_score_similarity_class)
    elif experiment_number == "3":  # SSIM as fitness - to see if the confidence increases with increasing SSIM
        return ssim_score_similarity_class


# Evolutionary algorithm to optimize images
def run_evolution(experiment_number, toolbox, ngen, model, input_shape, target_digit_for_confidence, target_digit_for_similarity, similarity_metric, output_subdir, generation_interval, replicate, target_image, image_for_similarity_comarison, model_name, dataset_name, seed):
    """
    Run the evolutionary algorithm to optimize images.
    """
    start_time = time.time()
    early_stop_gen = None

    os.makedirs(output_subdir, exist_ok=True)
    batch_dir = os.path.join(output_subdir, "batch_image_grid")
    os.makedirs(batch_dir, exist_ok=True)

    hdf5_path = os.path.join(output_subdir, "evolved_images.hdf5")
    initialize_hdf5_file(hdf5_path, image_shape=input_shape)



    population = toolbox.population(n=50)

    # Lists to track generation-wise metrics
    generation_images = []
    generation_confidences_for_target = []
    generation_confidences_for_similarity = []
    generation_ssims_for_target = []
    generation_ssims_for_similarity = []
    generation_nccs_for_target = []
    generation_nccs_for_similarity = []
    generation_indices = []

    # CSV file to save scores
    scores_csv_path = os.path.join(output_subdir, f'scores_digit_{target_digit_for_confidence}_rep{replicate}.csv')
    with open(scores_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Initialize tqdm progress bar
        progress_bar = tqdm(range(ngen + 1), desc="Evolving Generations", unit="gen", leave=False)

        try:
            for gen in progress_bar:
                # 1. Selection: Select individuals for the next generation
                offspring = toolbox.select(population, len(population))

                # 2. Cloning: Clone the selected individuals for modification
                offspring = list(map(toolbox.clone, offspring))

                # 3. Crossover: Apply crossover to pairs of offspring
                for child1, child2 in zip(offspring[::2], offspring[1::2]):
                    if np.random.rand() < 0.5:  # Apply crossover with 50% probability
                        toolbox.mate(child1, child2)
                        del child1.fitness.values  # Invalidate fitness of child1
                        del child2.fitness.values  # Invalidate fitness of child2

                # 4. Mutation: Mutate offspring with a certain probability
                for mutant in offspring:
                    if np.random.rand() < 0.2:  # Mutation probability
                        toolbox.mutate(mutant)
                        del mutant.fitness.values  # Invalidate fitness of mutant

                # 5. Repair: Clamp values to ensure valid bounds
                for ind in offspring:
                    for i in range(len(ind)):
                        ind[i] = np.clip(ind[i], 0, 1)

                # 6. Evaluation: Evaluate the fitness of individuals with invalid fitness
                invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
                for ind in invalid_ind:
                    confidence_score_for_target_digit, confidence_score_for_similarity_digit, ssim_score_target_class, ssim_score_similarity_class, ncc_score_target_class, ncc_score_similarity_class, confidence_scores = collect_scores(
                        ind, model, input_shape, target_image, image_for_similarity_comarison, target_digit_for_confidence, target_digit_for_similarity
                    )
                    fitness_score = fitness(experiment_number, confidence_score_for_target_digit, ssim_score_similarity_class, confidence_score_for_similarity_digit)
                    ind.fitness.values = (fitness_score,)

                # 7. Replacement: Replace the old population with the new offspring
                population[:] = offspring

                # 8. Get the best individual of the generation
                best_ind = tools.selBest(population, 1)[0]  # Select the best individual
                best_image = np.array(best_ind).reshape(input_shape)
                confidence_score_for_target_digit, confidence_score_for_similarity_digit, ssim_score_target_class, ssim_score_similarity_class, ncc_score_target_class, ncc_score_similarity_class, confidence_scores = collect_scores(
                    best_ind, model, input_shape, target_image, image_for_similarity_comarison, target_digit_for_confidence, target_digit_for_similarity
                )

                fitness_score = fitness(experiment_number, confidence_score_for_target_digit, ssim_score_similarity_class, confidence_score_for_similarity_digit)

                # Save scores to CSV
                if gen == 0:
                    # Write header with dynamic confidence score columns
                    header = ["Generation", "Fitness", "Confidence (Target Class)", "Confidence (Similarity Class)", "SSIM (Target Class)", "SSIM (Similarity Class)", "SSIM (Target Class)", "SSIM (Similarity Class)"] + [f"Confidence_Class_{i}" for i in range(len(confidence_scores))]
                    writer.writerow(header)

                # Write row with dynamic confidence score columns
                row = [gen, fitness_score, confidence_score_for_target_digit, confidence_score_for_similarity_digit, ssim_score_target_class, ssim_score_similarity_class, ncc_score_target_class, ncc_score_similarity_class] + list(confidence_scores)
                writer.writerow(row)
                file.flush()

                # 9. Track metrics and visualize results at intervals
                
                if gen <= 5 * generation_interval: # 500 for sklearn dataset, 5000 for mnist dataset
                    interval = generation_interval / 20 # 5 for sklearn dataset, 50 for mnist dataset
                else:
                    interval = generation_interval

                if gen % interval == 0 or gen == ngen or fitness_score >= 0.9999:
                    generation_images.append(best_image)
                    generation_confidences_for_target.append(confidence_score_for_target_digit)
                    generation_confidences_for_similarity.append(confidence_score_for_similarity_digit)
                    generation_ssims_for_target.append(ssim_score_target_class)
                    generation_ssims_for_similarity.append(ssim_score_similarity_class)
                    generation_nccs_for_target.append(ncc_score_target_class)
                    generation_nccs_for_similarity.append(ncc_score_similarity_class)
                    generation_indices.append(gen)

                    render_images_in_batches(
                        generation_images,
                        generation_confidences_for_target,
                        generation_confidences_for_similarity,
                        generation_ssims_for_target,
                        generation_ssims_for_similarity,
                        generation_nccs_for_target,
                        generation_nccs_for_similarity,
                        generation_indices,
                        batch_dir,
                        target_digit_for_confidence,
                        target_digit_for_similarity,
                        replicate,
                    )

                    plot_scores_vs_generations(
                        generation_indices,
                        generation_confidences_for_target,
                        generation_confidences_for_similarity,
                        generation_ssims_for_target,
                        generation_ssims_for_similarity,
                        generation_nccs_for_target,
                        generation_nccs_for_similarity,
                        output_subdir,
                        target_digit_for_confidence,
                        target_digit_for_similarity,
                        replicate,
                        model_name,
                        dataset_name,
                    )

                    append_to_hdf5(
                        file_path=hdf5_path,
                        gen=gen,  # Generation count
                        image=best_image,
                        fitness=fitness_score,
                        confidence_target=confidence_score_for_target_digit,
                        confidence_similarity=confidence_score_for_similarity_digit,
                        ssim_target=ssim_score_target_class,
                        ssim_similarity=ssim_score_similarity_class,
                        ncc_target=ncc_score_target_class,
                        ncc_similarity=ncc_score_similarity_class
                    )

                # Optional: Terminate if fitness reaches a threshold
                if fitness_score >= 0.9999:
                    tqdm.write(f"Stopping early at generation {gen} as fitness threshold reached.")
                    early_stop_gen = gen
                    break

        finally:
        # Extract essential tqdm details
            tqdm_details = {
                "current": progress_bar.n,  # Current progress
                "total": progress_bar.total,  # Total generations
                "rate": progress_bar.format_dict["rate"],  # Generations per second
                "elapsed_time": progress_bar.format_dict["elapsed"],  # Elapsed time
            }
            progress_bar.close()

     # Total runtime and save summary
    total_time = time.time() - start_time
    experiment_details = {
        "experiment_number": experiment_number,
        "target_digit_for_confidence": target_digit_for_confidence,
        "target_digit_for_similarity": target_digit_for_similarity,
        "similarity_metric": similarity_metric,
        "model_name": model_name,
        "dataset_name": dataset_name,
        "replicate": replicate,
        "generation_interval": generation_interval,
    }
    save_run_summary(output_subdir, early_stop_gen, experiment_details, tqdm_details)

    print(f"Run completed in {total_time:.2f} seconds.")
    print(f"Output saved to: {output_subdir}")
