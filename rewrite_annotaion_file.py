import pandas as pd
import os

def rewrite_annotation_file(input_csv_path, output_csv_path):
    """
    Rewrites an annotation CSV file, replacing the 'img_' prefix in the
    'image_name' column with 'stars-'.

    Args:
        input_csv_path (str): Path to the input annotation CSV file.
        output_csv_path (str): Path to save the rewritten annotation CSV file.
    """
    try:
        # Read the input CSV file into a pandas DataFrame
        df = pd.read_csv(input_csv_path)

        # Check if the 'image_name' column exists
        if 'image_name' not in df.columns:
            print(f"Error: 'image_name' column not found in {input_csv_path}")
            return

        # Replace 'img_' with 'stars-' in the 'image_name' column
        df['image_name'] = df['image_name'].str.replace('img_', 'stars-', regex=False)

        # Save the modified DataFrame to a new CSV file
        df.to_csv(output_csv_path, index=False)

        print(f"Successfully rewrote '{input_csv_path}' and saved the result to '{output_csv_path}'")

    except FileNotFoundError:
        print(f"Error: Input file '{input_csv_path}' not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    input_file = "/home/student/image_labels.csv"  # Replace with the actual path to your annotation file
    output_file = "/home/student/re_image_labels.csv" # Replace with the desired path for the new file

    # Example usage:
    rewrite_annotation_file(input_file, output_file)

    # You can then update the 'annotation_file_path' variable in your
    # training script to point to the 'output_file'.
