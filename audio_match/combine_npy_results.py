import numpy as np
import sys

def main():
    if len(sys.argv) < 3:
        print("Usage: python combine_npy_results.py input1.npy input2.npy ... output.npy")
        sys.exit(1)
    input_files = sys.argv[1:-1]
    output_file = sys.argv[-1]

    all_results = []
    for npy_file in input_files:
        data = np.load(npy_file, allow_pickle=True)
        all_results.extend(data)

    np.save(output_file, all_results)
    print(f"Saved combined results to {output_file}")

if __name__ == '__main__':
    main()
