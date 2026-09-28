import numpy as np

def display_npz_contents(file_path):
    global npz_data  # Declare as global so it appears in Variable Explorer
    try:
        # Load the .npz file
        data = np.load(file_path, allow_pickle=True)  # Allow complex objects

        npz_data = {}  # Dictionary to store arrays

        print(f"Contents of {file_path}:")
        print("=" * 50)

        # Iterate over each array in the file
        for key in data.files:
            array = data[key]
            npz_data[key] = array  # Store arrays globally
            print(f"Variable Name: {key}")
            print(f"  Shape: {array.shape}")
            print(f"  Data Type: {array.dtype}")
            print(f"  Complete Matrix:\n{array}")
            print("=" * 50)

        data.close()

    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        npz_data = None  # Assign None in case of an error

# Set your file path here
file_path = "E:/Ph_D_2020/Video_Summarization/Implementations/QDS-DETR/features/clip_text_features/qid8737.npz"  # Change this to your actual file path

# Call the function with the file path
display_npz_contents(file_path)
