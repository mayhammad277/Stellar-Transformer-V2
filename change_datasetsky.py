import os
import shutil

# Define the path to your dataset folder
dataset_folder = '/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated/'

# Define the path where you want to create the class folders
output_folder = '/media/student/B076126976123098/my_data/SiT/dataset_sky/sky_with_angles7_guassian_global_centroided_processed_with_red_centroid_grs_avoided_centroid_rotated_folder/'

# Ensure the output folder exists
os.makedirs(output_folder, exist_ok=True)

# Iterate over all files in the dataset folder
for filename in os.listdir(dataset_folder):
    print(filename)
    if filename.endswith('.png'):
        # Extract the class number from the filename
        class_number = filename.split('_')[0][1:]  # Extracts the number after 'N' (e.g., '0', '5', etc.)
        
        # Create the class folder name in the format 'N+num'
        class_folder_name = f'N{class_number}'
        
        # Create the class folder if it doesn't exist
        class_folder = os.path.join(output_folder, class_folder_name)
        os.makedirs(class_folder, exist_ok=True)
        
        # Copy the image to the corresponding class folder
        shutil.copy(os.path.join(dataset_folder, filename), os.path.join(class_folder, filename))

print("Dataset organized into class folders with images copied.")
