#!/bin/bash

# Script to delete folders with names N_Ni where i is NOT divisible by 30

# Safety first - dry run mode
echo "DRY RUN - Showing folders that would be deleted:"
find . -type d -name "N_N*" | while read -r folder; do
    # Extract the number part after N_N
    num=$(basename "$folder" | awk -F'N_N' '{print $2}')
    
    # Check if it's a number NOT divisible by 30
    if [[ "$num" =~ ^[0-9]+$ ]] && (( num % 30 != 0 )); then
        echo "Would delete: $folder"
    fi
done

# Ask for confirmation
read -p "Do you want to actually delete these folders? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Deleting folders..."
    find . -type d -name "N_N*" | while read -r folder; do
        num=$(basename "$folder" | awk -F'N_N' '{print $2}')
        if [[ "$num" =~ ^[0-9]+$ ]] && (( num % 30 != 0 )); then
            echo "Deleting: $folder"
            rm -rf "$folder"
        fi
    done
    echo "Deletion complete."
else
    echo "Operation cancelled."
fi
