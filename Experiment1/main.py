import os
import argparse
import numpy as np
from data_processing import load_dataset, save_mean_median_images, load_median_image, evaluate_model_accuracy
from evolution import run_evolution, evaluate_model_with_foolability
from models import get_model, train_model
from deap import base, creator, tools

def main():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run machine learning models on various datasets.")
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., sklearnDigits, mnistDigits, mnistFashion, CIFAR10, CIFAR100)")
    parser.add_argument("model_name", type=str, help="Model name (e.g., SVM, RF, CNN, RNN)")
    parser.add_argument("target_digit", type=int, help="Target digit to train and evolve")
    parser.add_argument("gen_interval", type=int, help="Interval for saving generation images")
    parser.add_argument("replicate", type=int, help="Replicate index")
    parser.add_argument("ngen", type=int, help="Number of generations for evolution")
    args = parser.parse_args()

    # Extract values from arguments
    dataset_name = args.dataset_name
    model_name = args.model_name
    target_digit = args.target_digit
    generation_interval = args.gen_interval
    replicate = args.replicate
    ngen = args.ngen

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    X_train, X_test, y_train, y_test, input_shape, num_classes = load_dataset(dataset_name)

    # Dynamically set the mean/median image directory based on the dataset name
    mean_median_dir = os.path.join("mean_median_images", dataset_name)
    os.makedirs(mean_median_dir, exist_ok=True)

    # Save mean and median images using training data
    print("Saving mean and median images...")
    save_mean_median_images(X_train, y_train, mean_median_dir)

    # Train the selected model
    print("Training the model...")
    model = get_model(model_name, input_shape)
    trained_model = train_model(model, model_name, X_train, y_train)

    # Evaluate the model
    evaluate_model_accuracy(trained_model, model_name, X_train, y_train, X_test, y_test)

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
    output_dir = os.path.join("evolution_results", dataset_name, model_name)
    os.makedirs(output_dir, exist_ok=True)

    # Run evolution
    print("Running evolution...")
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
