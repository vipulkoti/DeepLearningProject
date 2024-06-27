# Filename:
#   main.py
# Description:
#   Main entry point to run experiments + generate graphs for report
#

from data_preprocessor import DataSplit, generate_dataset_splits, get_preprocessed_captions

def main():
    # Get dataset splits
    splits = generate_dataset_splits()

    # Get preprocessed captions
    captions = get_preprocessed_captions()
    k0 = list(captions.keys())[0]
    k1 = list(captions.keys())[1]
    print(f"Example caption: {captions[k0]}\n")
    print(f"Example caption: {captions[k1]}\n")

if __name__ == "__main__":
    main()