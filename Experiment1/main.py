import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from deap import base, creator, tools
from skimage.metrics import structural_similarity as ssim
from skimage.feature import match_template
from skimage.io import imread
import csv

# Define CNN and RNN models
def create_cnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def create_rnn_model(input_shape):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.SimpleRNN(50, return_sequences=True),
        tf.keras.layers.SimpleRNN(50),
        tf.keras.layers.Dense(100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# SSIM computation function
def compute_ssim(image1, image2):
    return ssim(image1, image2, data_range=1.0)

# NCC computation function
def compute_ncc(image1, image2):
    if image1.shape != image2.shape:
        raise ValueError("Images must be of the same dimensions")
    result = match_template(image1, image2)
    return result.max()

# Evaluation function with F = confidence - SSIM as fitness
def evaluate_model_with_foolability(individual, model, input_shape, median_image, target_digit):
    image = np.array(individual).reshape(*input_shape)
    ssim_score = compute_ssim(image, median_image)
    
    if isinstance(model, tf.keras.Model):
        image_expanded = np.array(individual).reshape(1, *input_shape, 1)
        probabilities = model.predict(image_expanded)
    else:
        image_expanded = np.array(individual).reshape(1, -1)
        probabilities = model.predict_proba(image_expanded)
    
    confidence_score = probabilities[0][target_digit]

    # foolability_score = -confidence_score + ssim_score


    foolability_score = confidence_score - abs(ssim_score)
    
    
    return foolability_score, confidence_score, ssim_score

# Function to render images in batches of up to 25
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
            axs[row, col].set_title(f'Gen {idx}\nConf: {conf:.9f}\nSSIM: {ssim:.9f} \nNCC: {ncc:.9f}', fontsize=8)
            axs[row, col].axis('off')

        for j in range(len(images[start_idx:end_idx]), 25):
            row, col = divmod(j, 5)
            axs[row, col].axis('off')

        plt.savefig(os.path.join(output_subdir, f"digit_{target_digit}_rep{replicate}_batch_{batch_idx + 1}.png"))
        plt.close(fig)

# Plotting scores over generations
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

# Save mean and median images for reference
def save_mean_median_images(X, y, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for digit in range(10):
        digit_images = X[np.argmax(y, axis=1) == digit]
        mean_image = np.mean(digit_images, axis=0)
        median_image = np.median(digit_images, axis=0)
        plt.imsave(os.path.join(output_dir, f'digit_{digit}_mean.png'), mean_image, cmap='gray')
        plt.imsave(os.path.join(output_dir, f'digit_{digit}_median.png'), median_image, cmap='gray')

# Function to load the median image for a given digit
def load_median_image(output_dir, digit):
    median_image_path = os.path.join(output_dir, f'digit_{digit}_median.png')
    return imread(median_image_path, as_gray=True)

# Evolutionary algorithm to optimize images using F as fitness
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

# Command-line argument parsing
parser = argparse.ArgumentParser(description='Run machine learning models on digits dataset.')
parser.add_argument('model_name', type=str, help='Model name (e.g., SVM, RF, CNN, RNN)')
parser.add_argument('target_digit', type=int, help='Target digit to train and evolve')
parser.add_argument('gen_interval', type=int, help='Interval for saving generation images')
parser.add_argument('replicate', type=int, help='Replicate index')
parser.add_argument('ngen', type=int, help='Number of generations for evolution')
args = parser.parse_args()

# Extract values from arguments
model_name = args.model_name
target_digit = args.target_digit
generation_interval = args.gen_interval
replicate = args.replicate
ngen = args.ngen

# Model selection
models = {
    'MLP': lambda: MLPClassifier(hidden_layer_sizes=(100,), max_iter=500),
    'GBM': lambda: GradientBoostingClassifier(),
    'SVM': lambda: SVC(gamma='scale', probability=True),
    'RF': lambda: RandomForestClassifier(),
    'CNN': create_cnn_model,
    'RNN': create_rnn_model
}

# Load data
digits = datasets.load_digits()
X = digits.images / 16.0
y = np.eye(10)[digits.target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure mean and median images are saved
mean_median_dir = 'mean_median_images'
save_mean_median_images(X_train, y_train, mean_median_dir)

# Train selected model
input_shape = X_train[0].shape
if model_name in ['SVM', 'RF', 'GBM', 'MLP']:
    model = models[model_name]()
    model.fit(X_train.reshape((-1, 64)), np.argmax(y_train, axis=1))
    trained_model = model
elif model_name == "CNN":
    model = models[model_name]((8, 8, 1))
    model.fit(X_train[..., np.newaxis], y_train, epochs=10, batch_size=32, verbose=0)
    trained_model = model
elif model_name == "RNN":
    model = models[model_name]((8, 8))
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
    trained_model = model
else:
    raise ValueError(f"Unknown model name: {model_name}")

# Evolutionary setup
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


# creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# creator.create("Individual", list, fitness=creator.FitnessMin)


toolbox = base.Toolbox()
toolbox.register("attr_float", np.random.rand)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, input_shape[0] * input_shape[1])
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", lambda ind: evaluate_model_with_foolability(ind, trained_model, input_shape, load_median_image(mean_median_dir, target_digit), target_digit))

# Run evolution
output_dir = os.path.join('evolution_results', model_name)
run_evolution(toolbox, ngen=ngen, model=trained_model, input_shape=input_shape, target_digit=target_digit, output_subdir=output_dir, generation_interval=generation_interval, replicate=replicate, median_image=load_median_image(mean_median_dir, target_digit), model_name=model_name)
