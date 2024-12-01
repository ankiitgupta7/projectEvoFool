import os
import csv
import numpy as np
import tensorflow as tf  # Import TensorFlow
from deap import tools  # Import tools from DEAP
from metrics import compute_ncc, compute_ssim
from visualisation import render_images_in_batches, plot_scores_vs_generations

# Collect scores (confidence, SSIM, NCC)
def collect_scores(individual, model, input_shape, image_for_similarity_comarison, target_digit):
    """
    Compute confidence, SSIM, and NCC scores for a given individual.
    """
    image = np.array(individual).reshape(*input_shape)

    if not (0 <= image.min() <= image.max() <= 1):
        raise ValueError("Image values are not normalized.")
    if not (0 <= image_for_similarity_comarison.min() <= image_for_similarity_comarison.max() <= 1):
        raise ValueError("Median image values are not normalized.")

    # Compute SSIM
    ssim_score = compute_ssim(image, image_for_similarity_comarison)

    # Compute confidence
    if isinstance(model, tf.keras.Model):  # TensorFlow model check
        image_expanded = np.array(individual).reshape(1, *input_shape, 1)
        probabilities = model.predict(image_expanded)
    else:  # Sklearn or similar model check
        image_expanded = np.array(individual).reshape(1, -1)
        probabilities = model.predict_proba(image_expanded)

    confidence_score = probabilities[0][target_digit]

    # Compute NCC
    ncc_score = compute_ncc(image, image_for_similarity_comarison)

    return confidence_score, ssim_score, ncc_score


# Compute fitness based on scores
def fitness(experiment_number, confidence_score, ssim_score):
    """
    Compute the fitness of an individual based on the experiment number.
    """
    if experiment_number == "1":
        return confidence_score
    elif experiment_number == "2.1.1":
        return ssim_score - confidence_score
    elif experiment_number == "2.1.2":
        return (ssim_score + confidence_score)/2
    elif experiment_number == "2.2":
        return confidence_score - abs(ssim_score)
    elif experiment_number == "3":
        return ssim_score


# Evolutionary algorithm to optimize images
def run_evolution(experiment_number, toolbox, ngen, model, input_shape, target_digit_for_confidence, similarity_metric, output_subdir, generation_interval, replicate, image_for_similarity_comarison, model_name):
    """
    Run the evolutionary algorithm to optimize images.
    """
    os.makedirs(output_subdir, exist_ok=True)
    population = toolbox.population(n=50)

    # Lists to track generation-wise metrics
    generation_images = []
    generation_confidences = []
    generation_ssims = []
    generation_nccs = []
    generation_indices = []

    # CSV file to save scores
    scores_csv_path = os.path.join(output_subdir, f'scores_digit_{target_digit_for_confidence}_rep{replicate}.csv')
    with open(scores_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Generation", "Fitness", "Confidence", "SSIM", "NCC"])

        for gen in range(ngen + 1):
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
                confidence_score, ssim_score, ncc_score = collect_scores(
                    ind, model, input_shape, image_for_similarity_comarison, target_digit_for_confidence
                )
                fitness_score = fitness(experiment_number, confidence_score, ssim_score)
                ind.fitness.values = (fitness_score,)

            # 7. Replacement: Replace the old population with the new offspring
            population[:] = offspring

            # 8. Get the best individual of the generation
            best_ind = tools.selBest(population, 1)[0]  # Select the best individual
            best_image = np.array(best_ind).reshape(input_shape)
            confidence_score, ssim_score, ncc_score = collect_scores(
                best_ind, model, input_shape, image_for_similarity_comarison, target_digit_for_confidence
            )
            
            fitness_score = fitness(experiment_number, confidence_score, ssim_score)

            # Save scores to CSV
            writer.writerow([gen, fitness_score, confidence_score, ssim_score, ncc_score])
            file.flush()

            # 9. Track metrics and visualize results at intervals
            if gen % generation_interval == 0 or gen == ngen:
                generation_images.append(best_image)
                generation_confidences.append(confidence_score)
                generation_ssims.append(ssim_score)
                generation_nccs.append(ncc_score)
                generation_indices.append(gen)

                render_images_in_batches(
                    generation_images,
                    generation_confidences,
                    generation_ssims,
                    generation_nccs,
                    generation_indices,
                    output_subdir,
                    target_digit_for_confidence,
                    replicate,
                )

                plot_scores_vs_generations(
                    generation_indices,
                    generation_confidences,
                    generation_ssims,
                    generation_nccs,
                    output_subdir,
                    target_digit_for_confidence,
                    replicate,
                    model_name,
                )

            # Optional: Terminate if fitness reaches a threshold
            if fitness_score >= 0.9999:
                break
