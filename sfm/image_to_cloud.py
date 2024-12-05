import os
import shutil
import subprocess
import logging

def setup_project(image_dir, project_dir):
    """
    Sets up the OpenSFM project directory by copying images to the project.

    Parameters:
    image_dir (str): Path to the directory containing input images.
    project_dir (str): Path to the OpenSFM project directory.
    """
    if not os.path.exists(project_dir):
        os.makedirs(project_dir)

    # Copy images to the OpenSFM project directory
    images_dest_dir = os.path.join(project_dir, "images")
    if os.path.exists(images_dest_dir):
        shutil.rmtree(images_dest_dir)
    shutil.copytree(image_dir, images_dest_dir)

def run_opensfm_command(project_dir, command):
    """
    Runs an OpenSFM command on the given project directory.

    Parameters:
    project_dir (str): Path to the OpenSFM project directory.
    command (str): The OpenSFM command to run.
    """
    try:
        subprocess.run(["/home/christian/Downloads/Software/opensfm/OpenSfM/bin/opensfm", command, project_dir], check=True)
        logging.info(f"OpenSFM command '{command}' completed successfully.")
    except subprocess.CalledProcessError as e:
        logging.error(f"OpenSFM command '{command}' failed with error: {e}")
        raise

def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Paths (update these to your specific paths)
    image_dir = "/home/christian/Downloads/sfm_sandbox/sfm_runs/images"  # Directory with your input images
    project_dir = "/home/christian/Downloads/sfm_sandbox/sfm_runs/output"  # Directory where the OpenSFM project will be created

    try:
        # Step 1: Setup project
        setup_project(image_dir, project_dir)
        logging.info("Project setup complete.")

        # Step 2: Extract metadata from images
        run_opensfm_command(project_dir, "extract_metadata")

        # Step 3: Detect features in images
        run_opensfm_command(project_dir, "detect_features")

        # Step 4: Match features between images
        run_opensfm_command(project_dir, "match_features")

        # Step 4.5: Create Tracks
        run_opensfm_command(project_dir, "create_tracks")

        # Step 5: Create reconstruction
        run_opensfm_command(project_dir, "reconstruct")

        # Step 6: Undistort images
        run_opensfm_command(project_dir, "undistort")

        # Step 7: Compute depth maps
        run_opensfm_command(project_dir, "compute_depthmaps")

        # Step 8: Export the point cloud as a PLY file
        run_opensfm_command(project_dir, "export_ply")

        logging.info("Point cloud generation complete. Check the 'dense' folder in your project directory.")

    except Exception as e:
        logging.error(f"An error occurred: {e}")

    notif='.ply file generated'

    return notif

if __name__ == "__main__":
    main()
