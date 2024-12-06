import os
import sys
import subprocess
from glob import glob
from tqdm import tqdm

def main():
    ROOT_DIR = "output"

    PIPELINE_OUTPUT_DIR = "pipeline_output"
    os.makedirs(PIPELINE_OUTPUT_DIR, exist_ok=True)

    CUMULATIVE_RESULT_TXT = os.path.join(PIPELINE_OUTPUT_DIR, "cumulative_results.txt")
    CUMULATIVE_RESULT_NPY = os.path.join(PIPELINE_OUTPUT_DIR, "cumulative_results.npy")

    # Remove existing cumulative result files
    if os.path.exists(CUMULATIVE_RESULT_TXT):
        os.remove(CUMULATIVE_RESULT_TXT)
    if os.path.exists(CUMULATIVE_RESULT_NPY):
        os.remove(CUMULATIVE_RESULT_NPY)

    # Initialize an empty list to collect npy results
    cumulative_npy_results = []

    # Collect all runs
    runs = []
    for solver_dir in glob(os.path.join(ROOT_DIR, "*")):
        if os.path.isdir(solver_dir):
            solver_name = os.path.basename(solver_dir)
            for sound_group_dir in glob(os.path.join(solver_dir, "*")):
                if os.path.isdir(sound_group_dir):
                    sound_group_name = os.path.basename(sound_group_dir)
                    for sound_dir in glob(os.path.join(sound_group_dir, "*")):
                        if os.path.isdir(sound_dir):
                            sound_name = os.path.basename(sound_dir)
                            runs.append((solver_name, sound_group_name, sound_name, sound_dir))

    # Process each run with a progress bar
    for run in tqdm(runs, desc="Processing runs"):
        solver_name, sound_group_name, sound_name, sound_dir = run
        print(f"\nProcessing Solver: {solver_name}, Sound Group: {sound_group_name}, Sound: {sound_name}")

        # Define paths
        input_folder = sound_dir
        output_npz = os.path.join(sound_dir, "output.npz")
        trajectory_npy = os.path.join(sound_dir, "trajectory.npy")
        image_dir = input_folder
        detection_output = os.path.join(sound_dir, "detection_results")
        output_dir = os.path.join(sound_dir, "output")

        # Clear and create necessary directories
        # if os.path.exists(detection_output):
        #     subprocess.run(["rm", "-rf", detection_output])
        os.makedirs(detection_output, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)

        # Extract trajectories from output.npz
        subprocess.run(["python3", "extract_trajectories.py", output_npz, trajectory_npy])

        # Run object detection
        print("Running object detection...")
        subprocess.run([
            "python3", "scenic/detection_inference2.py",
            image_dir, detection_output, trajectory_npy, "--subsample_factor", "7"
        ])

        # Find the corresponding audio file
        audio_file = ""
        audio_search_path = os.path.join("input", sound_group_name)
        for root, dirs, files in os.walk(audio_search_path):
            for file in files:
                if file.endswith(".wav") and sound_name in file:
                    audio_file = os.path.join(root, file)
                    break
            if audio_file:
                break

        if not audio_file:
            print(f"Audio file not found for Sound: {sound_name} under input/{sound_group_name}")
            continue

        # Run multimodal analysis
        print("Running multimodal analysis...")
        cropped_dir = os.path.join(detection_output, "cropped_objects")
        detection_txt = os.path.join(detection_output, "detection_results.txt")
        subprocess.run([
            "python3", "multimodal/image_inference2.py",
            cropped_dir, detection_txt, audio_file, output_dir
        ])

        # Append results to cumulative result txt
        matching_results_txt = os.path.join(output_dir, "matching_results.txt")

        print("Saving cumulative results...")
        with open(CUMULATIVE_RESULT_TXT, "a") as f_out:
            f_out.write(f"Solver: {solver_name}, Sound Group: {sound_group_name}, Sound: {sound_name}\n")
            with open(matching_results_txt, "r") as f_in:
                f_out.write(f_in.read())
            f_out.write("\n")

        # Collect npy results
        matching_results_npy = os.path.join(output_dir, "matching_results.npy")
        if os.path.exists(matching_results_npy):
            cumulative_npy_results.append(matching_results_npy)

    # After all runs, combine npy results into cumulative_results.npy
    if cumulative_npy_results:
        print(f"Combining npy results into {CUMULATIVE_RESULT_NPY}")
        subprocess.run(["python3", "combine_npy_results.py"] + cumulative_npy_results + [CUMULATIVE_RESULT_NPY])
    else:
        print("No npy results to combine.")

    print(f"Processing completed. Cumulative results saved in {PIPELINE_OUTPUT_DIR}")

if __name__ == "__main__":
    main()
