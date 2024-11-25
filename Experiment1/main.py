import os
import argparse
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from data_processing import save_mean_median_images, load_median_image
from evolution import run_evolution, evaluate_model_with_foolability
from models import create_cnn_model, create_rnn_model
from deap import base, creator, tools

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

# Train the selected model
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
output_dir = os.path.join('evolution_results', model_name)
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
    model_name=model_name
)
