import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from evolution import run_evolution, evaluate_model_with_foolability
from unpickle import load_saved_dataset, load_median_image, load_best_image, load_trained_model
from deap import base, creator, tools


def run():
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

    # # Load dataset from pickle file
    # print(f"loading dataset: {dataset_name}")
    # X_train, X_test, y_train, y_test, input_shape, num_classes = load_saved_dataset(dataset_name)
    # print("Dataset loaded successfully:")
    # print(f"X_train shape: {X_train.shape}")
    # print(f"X_test shape: {X_test.shape}")
    # print(f"y_train shape: {y_train.shape}")
    # print(f"y_test shape: {y_test.shape}")
    # print(f"Input shape: {input_shape}")
    # print(f"Number of classes: {num_classes}")

    # # Load median image for the target digit
    # median_image = load_median_image(dataset_name, target_digit)
    # print(f"Median image for digit {target_digit} loaded successfully.")
    # print(f"Median image shape: {median_image.shape}")

    # # Load the trained model
    # trained_model = load_trained_model(dataset_name, model_name)
    # print(f"Trained model loaded successfully.")


    # Load dataset from pickle file
    print(f"Loading dataset: {dataset_name}")
    X_train, X_test, y_train, y_test, input_shape, num_classes = load_saved_dataset(dataset_name)
    print("Dataset loaded successfully:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")


    # Load median image for the target digit
    median_image = load_median_image(dataset_name, target_digit)
    print(f"Median image for digit {target_digit} loaded successfully.")
    print(f"Median image shape: {median_image.shape}")

    # Visualize the median image
    plt.imshow(median_image.squeeze(), cmap="gray")
    plt.title(f"Median Image for Digit {target_digit}")
    plt.show()

    # Validate median image
    assert median_image.shape == input_shape, "Median image shape mismatch!"

    # Load the best image for the target digit
    best_image_data = load_best_image(dataset_name, model_name, target_digit, "train")
    best_image = best_image_data["image"]
    confidence = best_image_data["confidence"]
    class_label = best_image_data["class"]

    print(f"Best image for class {class_label} loaded successfully.")
    print(f"Confidence score: {confidence}")
    print(f"Image shape: {best_image.shape}")

    # Visualize the best image
    plt.imshow(best_image.squeeze(), cmap="gray" if best_image.ndim == 2 else None)
    plt.title(f"Best Image for Class {class_label}\nConfidence: {confidence:.4f}")
    plt.axis("off")
    plt.show()

    
    image_for_similarity_comarison = best_image

    # Load the trained model
    trained_model = load_trained_model(dataset_name, model_name)
    print(f"Trained model loaded successfully.")


    # Evolutionary setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # Attribute generator for individuals
    toolbox.register("attr_float", np.random.rand)

    # Individual and population initialization
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, np.prod(input_shape))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operations
    toolbox.register("mate", tools.cxOnePoint)  # Register One-Point Crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Gaussian mutation
    toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection

    # Evaluation function
    toolbox.register("evaluate", lambda ind: evaluate_model_with_foolability(
        ind, trained_model, input_shape, image_for_similarity_comarison, target_digit))

    # Output directory for evolution results
    output_dir = os.path.join("data_generated", dataset_name, model_name)
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
        image_for_similarity_comarison=image_for_similarity_comarison,
        model_name=model_name,
    )


run()
