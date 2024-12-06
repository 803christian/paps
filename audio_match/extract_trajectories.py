import numpy as np
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: python extract_trajectories.py input.npz output.npy")
        sys.exit(1)
    input_npz = sys.argv[1]
    output_npy = sys.argv[2]

    data = np.load(input_npz, allow_pickle=True)
    # Use the correct label 'trajectory'
    if 'trajectory' in data:
        trajectory = data['trajectory']
        np.save(output_npy, trajectory)
        print(f"Saved trajectory to {output_npy}")
    else:
        print("Error: 'trajectory' key not found in the npz file.")
        sys.exit(1)

if __name__ == '__main__':
    main()
