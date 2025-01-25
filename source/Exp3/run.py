import os
import argparse
import random
import numpy as np
import tensorflow as tf
from evolution import run_evolution, collect_scores
from unpickle import load_saved_dataset, load_median_image, load_trained_model
from deap import base, creator, tools

def generate_seed(exp_number, dataset_name, similarity_metric, target_conf, target_sim, replicate):
    """
    Generate a unique seed based on experiment parameters.
    """
    exp_base = {"1": 100, "2_1a": 211, "2_1b": 212, "2_2": 220, "3": 300}
    dataset_base = {"sklearnDigits": 1, "mnistDigits": 2, "mnistFashion": 3, "CIFAR10": 4, "CIFAR100": 5}
    similarity_metric_base = {"SSIM": 1, "NCC": 2, "LPIPS": 3, "PSNR": 4}

    exp_value = exp_base.get(exp_number, 0)
    dataset_value = dataset_base.get(dataset_name, 0)
    sim_metric_value = similarity_metric_base.get(similarity_metric, 0)

    if exp_value == 0 or dataset_value == 0 or sim_metric_value == 0:
        raise ValueError("Unknown experiment, dataset, or similarity metric.")

    if not (0 <= target_conf <= 99) or not (0 <= target_sim <= 99):
        raise ValueError("Target class indices must be between 0 and 99.")

    seed = int(f"{exp_value}{dataset_value}{sim_metric_value}{target_conf:02d}{target_sim:02d}{replicate:02d}")
    return seed

def run():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Run machine learning models on various datasets.")
    parser.add_argument("experiment_number", type=str, help="Experiment number (e.g., 1, 2_1a, 2_1b, 2_2, 3)")
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., sklearnDigits, mnistDigits, mnistFashion, CIFAR10, CIFAR100)")
    parser.add_argument("target_class_for_confidence", type=int, help="Target class for model confidence")
    parser.add_argument("target_class_for_similarity", type=int, help="Target class for similarity comparison")
    parser.add_argument("similarity_metric", type=str, help="Similarity metric to use for evaluation (e.g., SSIM, NCC, LPIPS, PSNR)")
    parser.add_argument("gen_interval", type=int, help="Interval for saving generation images")
    parser.add_argument("replicate", type=int, help="Replicate index")
    parser.add_argument("ngen", type=int, help="Number of generations for evolution")

    args = parser.parse_args()

    # Extract arguments
    experiment_number = args.experiment_number
    dataset_name = args.dataset_name
    target_class_for_confidence = args.target_class_for_confidence
    target_class_for_similarity = args.target_class_for_similarity
    similarity_metric = args.similarity_metric
    generation_interval = args.gen_interval
    replicate = args.replicate
    ngen = args.ngen

    # Define model names
    model_names = ["RF", "XGB", "SVM", "MLP", "CNN", "RNN"]

    # Generate a unique seed for the experiment
    seed = generate_seed(experiment_number, dataset_name, similarity_metric, target_class_for_confidence, target_class_for_similarity, replicate)
    print(f"Seed Generated for this experiment: {seed}")

    # Set seeds for reproducibility
    np.random.seed(seed % (2**32))
    tf.random.set_seed(seed)
    random.seed(seed)

    # Load dataset
    print(f"Loading dataset: {dataset_name}")
    X_train, X_test, y_train, y_test, input_shape, num_classes = load_saved_dataset(dataset_name)
    print("Dataset loaded successfully:")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"Input shape: {input_shape}")
    print(f"Number of classes: {num_classes}")

    # Load median images
    median_image_similarity_class = load_median_image(dataset_name, target_class_for_similarity)
    median_image_target_class = load_median_image(dataset_name, target_class_for_confidence)

    print(f"Median image for similarity_class {target_class_for_similarity} loaded successfully.")
    print(f"Median similarity_class image shape: {median_image_similarity_class.shape}")
    print(f"Median image for target_class {target_class_for_confidence} loaded successfully.")
    print(f"Median target_class image shape: {median_image_target_class.shape}")

    # Load trained models
    trained_models = {}
    for model_name in model_names:
        trained_models[model_name] = load_trained_model(dataset_name, model_name)
        print(f"Trained model {model_name} loaded successfully.")

    # Evolutionary setup
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)
    toolbox = base.Toolbox()

    # Attribute generator
    toolbox.register("attr_float", lambda: random.uniform(0, 1))

    # Individual and population initialization
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, np.prod(input_shape))
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Genetic operations
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    # Evaluation function (SSIM as fitness)
    toolbox.register("evaluate", lambda ind: collect_scores(
        ind, trained_models, input_shape, median_image_target_class, median_image_similarity_class
    )[1])  # Use SSIM_Target as fitness

    # Output directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
    output_subdir = os.path.join(
        parent_dir,
        "data_generated",
        f"Exp_{experiment_number}",
        dataset_name,
        f"class_{target_class_for_confidence}",
        f"replicate_{replicate}"
    )
    os.makedirs(output_subdir, exist_ok=True)

    # Run evolution
    print("Running evolution...")
    run_evolution(
        experiment_number=experiment_number,
        toolbox=toolbox,
        ngen=ngen,
        trained_models=trained_models,
        input_shape=input_shape,
        target_class_for_confidence=target_class_for_confidence,
        target_class_for_similarity=target_class_for_similarity,
        similarity_metric=similarity_metric,
        output_subdir=output_subdir,
        generation_interval=generation_interval,
        replicate=replicate,
        target_image=median_image_target_class,
        image_for_similarity_comarison=median_image_similarity_class,
        model_names=model_names,
        dataset_name=dataset_name,
        seed=seed,
    )

if __name__ == "__main__":
    run()
