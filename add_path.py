import pandas as pd
import os

def create_full_path_annotation(annotation_file, image_dir, star_map_dir, output_csv_path):
    """
    Reads an annotation CSV, adds full paths to image names and creates
    a new column with full paths to corresponding star map files.

    Args:
        annotation_file (str): Path to the input annotation CSV file.
        image_dir (str): Path to the directory containing the image files.
        star_map_dir (str): Path to the directory containing the star map files.
        output_csv_path (str): Path to save the new annotation CSV file.
    """
    try:
        # Read the input CSV file into a pandas DataFrame
        df = pd.read_csv(annotation_file)

        # Check if the 'image_name' column exists
        if 'image_name' not in df.columns:
            print(f"Error: 'image_name' column not found in {annotation_file}")
            return

        # Create the full image path
        df['full_image_path'] = df['image_name'].apply(lambda x: os.path.join(image_dir, x))

        # Create the base name for star map files (remove extension)
        df['base_name'] = df['image_name'].apply(lambda x: os.path.splitext(x)[0])

        # Create the star map file name
        df['star_map_file'] = df['base_name'].apply(lambda x: f"{x}_star_map.json")

        # Create the full star map path
        df['full_star_map_path'] = df['star_map_file'].apply(lambda x: os.path.join(star_map_dir, x))

        # Remove the temporary 'base_name' and 'star_map_file' columns
        df = df.drop(columns=['base_name', 'star_map_file'])

        # Save the modified DataFrame to a new CSV file
        df.to_csv(output_csv_path, index=False)

        print(f"Successfully created full paths and star map paths in '{output_csv_path}'")

    except FileNotFoundError:
        print(f"Error: Input file '{annotation_file}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_annotation_file = "/home/student/star_tracker/image_labels.csv" # Replace
    output_annotation_file = "/home/student/star_tracker/image_labels.csv"  # Replace
    image_directory = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed/images"  # Replace
    star_map_directory = "/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_star_processed/star_maps"  # Replace

    create_full_path_annotation(
        input_annotation_file,
        image_directory,
        star_map_directory,
        output_annotation_file
    )

    # Now update your StarDataset to use 'full_image_path' and 'full_star_map_path'
    # instead of just 'image_name' and constructing paths within __getitem__.
