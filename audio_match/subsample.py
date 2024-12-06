import os
import shutil

def subsample_folder(input_folder, output_folder, subsample_factor):
    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    # Get a sorted list of files in the input folder
    files = sorted(os.listdir(input_folder))
    
    # Select every nth file
    subsampled_files = files[::subsample_factor]

    # Copy selected files to the output folder
    for file in subsampled_files:
        input_path = os.path.join(input_folder, file)
        output_path = os.path.join(output_folder, file)
        shutil.copy(input_path, output_path)

    print(f"Copied {len(subsampled_files)} files from {input_folder} to {output_folder}")

def subsample_depth_input(depth_input_folder, output_base_folder, subsample_factor=50):
    # Define the paths to the subfolders for RGB and depth images
    rgb_folder = os.path.join(depth_input_folder, "rgb_images")
    depth_folder = os.path.join(depth_input_folder, "depth_images")
    
    # Define the output base folder for the subsampled data
    os.makedirs(output_base_folder, exist_ok=True)
    
    # Define paths for the subsampled subfolders inside the output base folder
    rgb_output_folder = os.path.join(output_base_folder, "rgb_images")
    depth_output_folder = os.path.join(output_base_folder, "depth_images")
    
    # Subsample each folder
    subsample_folder(rgb_folder, rgb_output_folder, subsample_factor)
    subsample_folder(depth_folder, depth_output_folder, subsample_factor)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Linearly subsample RGB and depth image folders.")
    parser.add_argument("depth_input_folder", type=str, help="Path to the depth_input folder containing rgb_images and depth_images subfolders.")
    parser.add_argument("output_base_folder", type=str, help="Path to the output folder where subsamples will be saved, e.g., depth_input_subsamples.")
    parser.add_argument("--subsample_factor", type=int, default=7, help="Factor by which to subsample the folders (default is 50).")
    
    args = parser.parse_args()
    subsample_depth_input(args.depth_input_folder, args.output_base_folder, args.subsample_factor)
