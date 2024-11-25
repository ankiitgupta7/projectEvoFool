import os
import argparse
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from data_processing import save_mean_median_images, load_median_image
from evolution import run_evolution, evaluate_model_with_foolability
from models import get_model, train_model
from deap import base, creator, tools

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run machine learning models on digits dataset.")
    parser.add_argument("model_name", type=str, help="Model name (e.g., SVM, RF, CNN, RNN)")
    parser.add_argument("target_digit", type=int, help="Target digit to train and evolve")
    parser.add_argument("gen_interval", type=int, help="Interval for saving generation images")
    parser.add_argument("replicate", type=int, help="Replicate index")
    parser.add_argument("ngen", type=int, help="Number of generations for evolution")
    args = parser.parse_args()

    # Extract values from arguments
    model_name = args.model_name
    target_digit = args.target_digit
    generation_interval = args.gen_interval
    replicate = args.replicate
    ngen = args.ngen

    # Load data
    digits = datasets.load_digits()
    X = digits.images / 16.0
    y = np.eye(10)[digits.target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Ensure mean and median images are saved
    mean_median_dir = "mean_median_images"
    save_mean_median_images(X_train, y_train, mean_median_dir)

    # Get and train the model
    input_shape = X_train[0].shape
    model = get_model(model_name, input_shape)
    trained_model = train_model(model, model_name, X_train, y_train)

    # Evolutionary setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()
    toolbox.register("attr_float", np.random.rand)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, input_shape[0] * input_shape[1])
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)
    toolbox.register("evaluate", lambda ind: evaluate_model_with_foolability(
        ind, trained_model, input_shape, load_median_image(mean_median_dir, target_digit), target_digit))

    # Output directory for evolution results
    output_dir = os.path.join("evolution_results", model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Run evolution
    run_evolution(
        toolbox=toolbox,
        ngen=ngen,
        model=trained_model,
        input_shape=input_shape,
        target_digit=target_digit,
        output_subdir=output_dir,
        generation_interval=generation_interval,
        replicate=replicate,
        median_image=load_median_image(mean_median_dir, target_digit),
        model_name=model_name,
    )

# Entry point for the script
if __name__ == "__main__":
    main()
