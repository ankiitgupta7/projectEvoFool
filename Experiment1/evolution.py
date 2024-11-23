from deap import base, creator, tools
import numpy as np
from metrics import compute_ssim, compute_ncc

def run_evolution(toolbox, ngen, model, input_shape, target_digit, output_subdir, generation_interval, replicate, median_image, model_name):
    os.makedirs(output_subdir, exist_ok=True)
    population = toolbox.population(n=100)
    generation_images = []
    generation_confidences = []
    generation_ssims = []
    generation_nccs = []
    generation_indices = []

    scores_csv_path = os.path.join(output_subdir, f'scores_digit_{target_digit}_rep{replicate}.csv')
    with open(scores_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Generation", "Foolability", "Confidence", "SSIM", "NCC"])

        for gen in range(ngen + 1):
            offspring = toolbox.select(population, len(population))
            offspring = list(map(toolbox.clone, offspring))

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < 0.5:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if np.random.rand() < 0.2:
                    toolbox.mutate(mutant)
                    del mutant.fitness.values

            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, (foolability_score, confidence_score, ssim_score) in zip(invalid_ind, fitnesses):
                ind.fitness.values = (foolability_score,)

            population[:] = offspring
            best_ind = tools.selBest(population, 1)[0]
            best_image = np.array(best_ind).reshape(input_shape)
            foolability_score, confidence_score, ssim_score = evaluate_model_with_foolability(best_ind, model, input_shape, median_image, target_digit)
            ncc_score = compute_ncc(best_image, median_image)

            writer.writerow([gen, foolability_score, confidence_score, ssim_score, ncc_score])
            file.flush()

            if gen % generation_interval == 0 or gen == ngen:
                generation_images.append(best_image)
                generation_confidences.append(confidence_score)
                generation_ssims.append(ssim_score)
                generation_nccs.append(ncc_score)
                generation_indices.append(gen)

                render_images_in_batches(generation_images, generation_confidences, generation_ssims, generation_nccs, generation_indices, output_subdir, target_digit, replicate)
                plot_scores_vs_generations(generation_indices, generation_confidences, generation_ssims, generation_nccs, output_subdir, target_digit, replicate, model_name)

            if foolability_score >= 1.0:
                break