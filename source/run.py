import os
import argparse
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from evolution import run_evolution, fitness, collect_scores
from unpickle import load_saved_dataset, load_median_image, load_best_image, load_trained_model
from deap import base, creator, tools

# runs with the following example command: python run.py 3 sklearnDigits SVM 5 5 SSIM 50 300 10000000

def generate_seed(exp_number, dataset_name, similarity_metric, target_conf, target_sim, replicate):
    """
    Generate a unique seed based on experiment parameters.

    Parameters:
    - exp_number (str): Experiment number (e.g., "1", "2_1a", "3").
    - dataset_name (str): Dataset name (e.g., "sklearnDigits", "mnistDigits").
    - similarity_metric (str): Similarity metric (e.g., "SSIM", "NCC").
    - target_conf (int): Target digit for confidence (0-99).
    - target_sim (int): Target digit for similarity (0-99).
    - replicate (int): Replicate number (1-30).

    Returns:
    - int: A unique seed for the given parameters.
    """
    # Experiment base values
    exp_base = {
        "1": 100,
        "2_1a": 211,
        "2_1b": 212,
        "2_2": 220,
        "3": 300
    }

    # Dataset base values (single-digit representation)
    dataset_base = {
        "sklearnDigits": 1,
        "mnistDigits": 2,
        "mnistFashion": 3,
        "CIFAR10": 4,
        "CIFAR100": 5
    }

    # Similarity metric values
    similarity_metric_base = {
        "SSIM": 1,
        "NCC": 2,
        "LPIPS": 3,
        "PSNR": 4
    }

    # Get base values
    exp_value = exp_base.get(exp_number, 0)
    dataset_value = dataset_base.get(dataset_name, 0)
    sim_metric_value = similarity_metric_base.get(similarity_metric, 0)

    if exp_value == 0:
        raise ValueError(f"Unknown experiment number: {exp_number}")
    if dataset_value == 0:
        raise ValueError(f"Unknown dataset name: {dataset_name}")
    if sim_metric_value == 0:
        raise ValueError(f"Unknown similarity metric: {similarity_metric}")
    if target_conf < 0 or target_conf > 99:
        raise ValueError(f"Target digit for confidence must be between 0 and 99, got {target_conf}")
    if target_sim < 0 or target_sim > 99:
        raise ValueError(f"Target digit for similarity must be between 0 and 99, got {target_sim}")

    # Generate the seed
    seed = int(f"{exp_value}{dataset_value}{sim_metric_value}{target_conf:02d}{target_sim:02d}{replicate:02d}")
    return seed

def run():
    # Command-line argument parsing
    parser = argparse.ArgumentParser(description="Run machine learning models on various datasets.")
    parser.add_argument("experiment_number", type=str, help="Experiment number (e.g., 1, 2_1_1, 2_1_2, 2_2, 3)")
    parser.add_argument("dataset_name", type=str, help="Dataset name (e.g., sklearnDigits, mnistDigits, mnistFashion, CIFAR10, CIFAR100)")
    parser.add_argument("model_name", type=str, help="Model name (e.g., SVM, RF, CNN, RNN)")
    parser.add_argument("target_digit_for_confidence", type=int, help="Target digit for model confidence")
    parser.add_argument("target_digit_for_similarity", type=int, help="Target digit for similarity comparison")
    parser.add_argument("similarity_metric", type=str, help="Similarity metric to use for evaluation (e.g., SSIM, NCC, LPIPS, PSNR)")
    parser.add_argument("gen_interval", type=int, help="Interval for saving generation images")
    parser.add_argument("replicate", type=int, help="Replicate index")
    parser.add_argument("ngen", type=int, help="Number of generations for evolution")


    args = parser.parse_args()

    # Extract values from arguments
    experiment_number = args.experiment_number
    dataset_name = args.dataset_name
    model_name = args.model_name
    target_digit_for_confidence = args.target_digit_for_confidence
    target_digit_for_similarity = args.target_digit_for_similarity
    similarity_metric = args.similarity_metric
    generation_interval = args.gen_interval
    replicate = args.replicate
    ngen = args.ngen

    # Generate a unique seed for the experiment
    seed = generate_seed(experiment_number, dataset_name, similarity_metric, target_digit_for_confidence, target_digit_for_similarity, replicate)
    print(f"Seed Generated for this experiment: {seed}")

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
    median_image_similarity_class = load_median_image(dataset_name, target_digit_for_similarity)
    median_image_target_class = load_median_image(dataset_name, target_digit_for_confidence)

    print(f"Median image for similarity_class {target_digit_for_similarity} loaded successfully.")
    print(f"Median image shape: {median_image_similarity_class.shape}")


    # # Visualize the median image
    # plt.imshow(median_image_similarity_class.squeeze(), cmap="gray")
    # plt.title(f"Median Image for Digit {target_digit_for_similarity}")
    # plt.show()


    # Load the training image for which model has the highest confidence for the target digit 
    best_confidence_image_data = load_best_image(dataset_name, model_name, target_digit_for_similarity, "train")
    best_confidence_image = best_confidence_image_data["image"]
    confidence = best_confidence_image_data["confidence"]
    class_label = best_confidence_image_data["class"]

    print(f"Best confidence image for class {class_label} loaded successfully.")
    print(f"Confidence score: {confidence}")
    print(f"Image shape: {best_confidence_image.shape}")

    # # Visualize the best image
    # plt.imshow(best_confidence_image.squeeze(), cmap="gray")
    # plt.title(f"Best Image for Class {class_label}\nConfidence: {confidence:.4f}")
    # plt.axis("off")
    # plt.show()



    image_for_similarity_comarison = median_image_similarity_class
    target_image = median_image_target_class

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
    # toolbox.register("mate", tools.cxOnePoint)  # Register One-Point Crossover - not used 
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)  # Gaussian mutation
    toolbox.register("select", tools.selTournament, tournsize=3)  # Tournament selection


    # Evaluation function
    toolbox.register("evaluate", lambda ind: (
        lambda confidence_score_for_target_digit, confidence_score_for_similarity_digit, ssim_score, ncc_score, confidence_scores: (fitness(experiment_number, confidence_score_for_target_digit, ssim_score, confidence_score_for_similarity_digit),))(
            *collect_scores(ind, trained_model, input_shape, target_image, image_for_similarity_comarison, target_digit_for_confidence)
        )
    )

    current_dir = os.path.dirname(os.path.abspath(__file__)) # where run.py is located, i.e., "source" directory
    parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir)) # source's parent directory

    # Output directory for evolution results
    output_subdir = os.path.join(
        parent_dir,  # Move to parent directory
        "data_generated",
        f"Exp_{args.experiment_number}",
        args.dataset_name,
        args.model_name,
        f"class_{args.target_digit_for_confidence}",
        f"replicate_{args.replicate}"
    )
    os.makedirs(output_subdir, exist_ok=True)

    # Run evolution
    print("Running evolution...")
    run_evolution(
        experiment_number=experiment_number,
        toolbox=toolbox,
        ngen=ngen,
        model=trained_model,
        input_shape=input_shape,
        target_digit_for_confidence=target_digit_for_confidence,
        target_digit_for_similarity=target_digit_for_similarity,
        similarity_metric=similarity_metric,
        output_subdir=output_subdir,
        generation_interval=generation_interval,
        replicate=replicate,
        target_image = target_image, # median image for target digit - for which confidence is calculated
        image_for_similarity_comarison=image_for_similarity_comarison, # median image for similarity digit - for which similarity is calculated
        model_name=model_name,
        dataset_name=dataset_name,
        seed=seed,
    )


run()
