#!/usr/bin/env python3
import os
import re
import shutil
import argparse

def delete_non_30_folders(root_dir):
    """Delete folders named N_Ni where i is not divisible by 30"""
    if not os.path.isdir(root_dir):
        print(f"Error: {root_dir} is not a valid directory")
        return False

    # First pass: dry run
    print(f"DRY RUN - Showing folders in {root_dir} that would be deleted:")
    to_delete = []
    
    for folder in os.listdir(root_dir):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path) and folder.startswith('N_N'):
            # Extract the number part
            match = re.match(r'N_N(\d+)$', folder)
            if match:
                num = int(match.group(1))
                if num % 30 != 0:  # Not divisible by 30
                    to_delete.append(folder_path)
                    print(f"Would delete: {folder_path}")
    
    # Ask for confirmation
    if not to_delete:
        print("No matching folders found")
        return True
    
    response = input(f"\nDelete {len(to_delete)} folders? (y/n) ").strip().lower()
    if response != 'y':
        print("Operation cancelled")
        return False
    
    # Second pass: actual deletion
    print("\nDeleting folders...")
    for folder_path in to_delete:
        try:
            print(f"Deleting: {folder_path}")
            shutil.rmtree(folder_path)
        except Exception as e:
            print(f"Error deleting {folder_path}: {e}")
            return False
    
    print("Deletion complete")
    return True

if __name__ == "__main__":
    
    dire="/media/student/B076126976123098/my_data/SiT/dataset_sky/new_all_augmented_processed/organized _30/starmaps/"
    
    delete_non_30_folders(dire)
