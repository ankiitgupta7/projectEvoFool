import pickle
import os

def test_median_images_pkl(pkl_path):
    """
    Loads the median_images.pkl file and checks if class '0' is present.
    """
    if not os.path.exists(pkl_path):
        print(f"File not found: {pkl_path}")
        return
    
    with open(pkl_path, 'rb') as f:
        median_images = pickle.load(f)

    print("Found keys (classes) in median_images.pkl:", list(median_images.keys()))
    
    if 0 in median_images:
        print("✅ Class 0 is present in median_images.")
    else:
        print("❌ Class 0 is NOT present in median_images. "
              "Below are the classes that are present:")
        print(list(median_images.keys()))
    

if __name__ == "__main__":
    # Update the path to wherever your median_images.pkl file is located
    pkl_path = "/home/ankit-gupta/Work/Projects/Active/projectEvoFool/source/dataset_info/mnistFashion/aggregated_images/median/pkl/median_images.pkl"
    
    test_median_images_pkl(pkl_path)



# import pickle
# import os
# import numpy as np

# def explore_data(obj, indent=0, max_depth=5):
#     """
#     Recursively explore the contents of a Python object (e.g. dict, list, array).
    
#     Args:
#         obj (any): The Python object to explore.
#         indent (int): Used to control indentation when printing.
#         max_depth (int): Maximum depth of recursion to prevent infinite loops.
#     """
#     prefix = " " * indent

#     if max_depth < 0:
#         print(prefix + "Max depth reached, stop exploring deeper.")
#         return

#     # Print the type of the object
#     print(prefix + f"Type: {type(obj)}")

#     # If it's a dict, show keys and explore each key's value
#     if isinstance(obj, dict):
#         print(prefix + f"Dict with {len(obj)} keys:")
#         for k, v in obj.items():
#             print(prefix + f"  Key: {k}")
#             explore_data(v, indent=indent + 4, max_depth=max_depth - 1)

#     # If it's a list or tuple, show length and explore each element
#     elif isinstance(obj, (list, tuple)):
#         print(prefix + f"{type(obj).__name__} with length {len(obj)}")
#         for i, v in enumerate(obj):
#             print(prefix + f"  Index: {i}")
#             explore_data(v, indent=indent + 4, max_depth=max_depth - 1)

#     # If it's a numpy array, show shape and dtype
#     elif isinstance(obj, np.ndarray):
#         print(prefix + f"ndarray with shape {obj.shape} and dtype {obj.dtype}")

#     # For any other type, just print the representation (up to some truncation)
#     else:
#         obj_repr = repr(obj)
#         # Trim if representation is too long
#         if len(obj_repr) > 100:
#             obj_repr = obj_repr[:97] + "..."
#         print(prefix + f"Value: {obj_repr}")


# def inspect_median_images_pkl(pkl_path, max_depth=3):
#     """
#     Load a pickle file and explore its contents.
#     """
#     if not os.path.exists(pkl_path):
#         print(f"File not found: {pkl_path}")
#         return
    
#     print(f"\nLoading pickle file: {pkl_path}\n")
#     with open(pkl_path, 'rb') as f:
#         data = pickle.load(f)
    
#     print("Exploring structure of the loaded data...\n")
#     explore_data(data, indent=0, max_depth=max_depth)


# if __name__ == "__main__":
#     # Change this path to your median_images.pkl or another pickle file.
#     pkl_path = "/home/ankit-gupta/Work/Projects/Active/projectEvoFool/source/dataset_info/mnistFashion/aggregated_images/median/pkl/median_images.pkl"
    
#     # Run inspection
#     inspect_median_images_pkl(pkl_path, max_depth=3)
