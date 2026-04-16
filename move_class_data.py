import os
import shutil
from tqdm import tqdm

def organize_files_by_class(input_dir, output_base_dir, file_type):
    """
    Organizes files into class folders based on their naming pattern N_X_i.ext
    
    Args:
        input_dir: Directory containing files to organize
        output_base_dir: Base directory where class folders will be created
        file_type: Type of files ('images', 'starmaps', or 'processed')
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_base_dir, exist_ok=True)
    
    # Get all files in input directory
    files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg', '.json'))]
    
    for filename in tqdm(files, desc=f"Organizing {file_type}"):
    
            # Extract class name from filename (N_X_i.ext)
            parts = filename.split('_')
            print("parts",parts)
            if len(parts) >= 2 and  'N'  in parts[0]:
                print("yes")
                class_num = parts[0]
                class_name = f"N_{class_num}"
                
                # Create class directory if it doesn't exist
                class_dir = os.path.join(output_base_dir, class_name)
                os.makedirs(class_dir, exist_ok=True)
                
                # Move file to its class directory
                src_path = os.path.join(input_dir, filename)
                dest_path = os.path.join(class_dir, filename)
                shutil.move(src_path, dest_path)
                


def main():
    # Configuration - update these paths to your actual directories
    base_dir = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/"
    
    # Input directories
    raw_images_dir = os.path.join(base_dir, "processed_images")
    starmaps_dir = os.path.join(base_dir, "star_maps")
    processed_dir = os.path.join(base_dir, "transformed_images")
    
    # Output directories
    organized_images_dir = os.path.join(base_dir, "organized_images")
    organized_starmaps_dir = os.path.join(base_dir, "organized_starmaps")
    organized_processed_dir = os.path.join(base_dir, "organized_processed")
    
    # Organize each type of file
    organize_files_by_class(raw_images_dir, organized_images_dir, "images")
    organize_files_by_class(starmaps_dir, organized_starmaps_dir, "star_maps")
    organize_files_by_class(processed_dir, organized_processed_dir, "processed")
    
    print("Organization complete!")

if __name__ == "__main__":
    main()
