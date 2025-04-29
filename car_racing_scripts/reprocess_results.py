import pickle
import matplotlib.pyplot as plt

# Load the saved results from the pickle file
def load_results(pickle_file):
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)
    return results

# Reprocess results and generate new plots
def reprocess_results(pickle_file, lr_filter=['5e-05', '00001', '00005', '00025', '0005', '001']):
    results = load_results(pickle_file)

    # Group results by epochs
    grouped_results = {}
    for mf, (mean, std) in results.items():
        if 'car_cnn_model_' in mf:
            label = mf.split('car_cnn_model_')[-1].split('.')[0]  # Extract label
            if '_epoch' in mf:
                epoch = mf.split('_epoch')[-1].split('_')[0]  # Extract epoch number
            if '_lr' in mf:
                lr = str(mf.split('_lr')[-1].split('.')[0])  # Extract learning rate
                print(lr)
                if lr_filter and lr not in lr_filter:
                    continue  # Skip if lr is not in the filter list
            else:
                print(f"Warning: Skipping file {mf} as it does not contain '_lr'")
                continue

            if epoch not in grouped_results:
                grouped_results[epoch] = []
            grouped_results[epoch].append((label, mean, std))
        else:
            print(f"Warning: Skipping file {mf} as it does not contain '_epoch'")

    # Generate new plots grouped by epochs
    for epoch, data in grouped_results.items():
        labels, means, stds = zip(*data)
        plt.figure(figsize=(10, 6))
        plt.bar(labels, means, yerr=stds, capsize=5, color='skyblue', edgecolor='black')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Average Score')
        plt.xlabel('Model Variants')
        plt.title(f'Model Performance Comparison (Epoch {epoch})')
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # plt.savefig(f'new_model_comparison_epoch_{epoch}.png')
        # print(f"Saved plot for Epoch {epoch} as 'new_model_comparison_epoch_{epoch}.png'")
        # Uncomment the line below to display the plot interactively
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reprocess evaluation results and generate new plots')
    parser.add_argument('--pickle_file', type=str, default='evaluation_results.pkl', help='Path to the pickle file')
    args = parser.parse_args()

    reprocess_results(args.pickle_file)