import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the saved results from the pickle file
def load_results(pickle_file):
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)
    return results

# Reprocess results and generate new plots grouped by batches
def reprocess_results(pickle_file, lr_filter=['5e-05', '00001', '00005', '00025', '0005', '001']):
    results = load_results(pickle_file)

    # Group results by batches
    grouped_results = {}
    for mf, (mean, std) in results.items():
        if 'car_cnn_model_' in mf:
            label = mf.split('car_cnn_model_')[-1].split('.')[0]  # Extract label

            # Extract epoch number
            if '_epoch' in mf:
                epoch = int(mf.split('_epoch')[-1].split('_')[0])  # Convert epoch to integer
            else:
                print(f"Warning: Skipping file {mf} as it does not contain '_epoch'")
                continue

            # Extract batch number
            if '_batch' in mf:
                batch = mf.split('_batch')[-1].split('_')[0]
            else:
                print(f"Warning: Skipping file {mf} as it does not contain '_batch'")
                continue

            # Extract learning rate
            if '_lr' in mf:
                lr = str(mf.split('_lr')[-1].split('.')[0])
                if lr_filter and lr not in lr_filter:
                    continue  # Skip if lr is not in the filter list
            else:
                print(f"Warning: Skipping file {mf} as it does not contain '_lr'")
                continue

            # Group by batch and epoch
            if batch not in grouped_results:
                grouped_results[batch] = {}
            if lr not in grouped_results[batch]:
                grouped_results[batch][lr] = {}
            if epoch not in grouped_results[batch][lr]:
                grouped_results[batch][lr][epoch] = []
            grouped_results[batch][lr][epoch].append((mean, std))  # Store mean and std
        else:
            print(f"Warning: Skipping file {mf} as it does not contain '_epoch' or '_batch'")

    # Generate new plots grouped by batches
    for batch, lrs in grouped_results.items():
        plt.figure(figsize=(12, 6))

        # Extract all unique epochs and assign consistent colors
        all_epochs = sorted({epoch for lr_data in lrs.values() for epoch in lr_data.keys()})  # Sort epochs numerically
        epoch_color_map = {epoch: color for epoch, color in zip(all_epochs, plt.cm.tab10.colors)}

        present_lrs = [lr for lr in lr_filter if lr in lrs]  # Filter learning rates present in the current batch
        lr_positions = np.arange(len(present_lrs))  # Positions for each learning rate group
        bar_width = 0.2  # Width of each bar
        offsets = np.arange(len(all_epochs)) * bar_width  # Offsets for epochs

        for i, lr in enumerate(present_lrs):  # Iterate only over present learning rates
            epochs = lrs[lr]
            for j, (epoch, data) in enumerate(sorted(epochs.items())):  # Sort epochs numerically
                means, stds = zip(*data)
                x_positions = lr_positions[i] + offsets[j]
                plt.bar(
                    x_positions,
                    means,
                    yerr=stds,
                    capsize=5,
                    width=bar_width,
                    color=epoch_color_map[epoch],  # Use consistent color for each epoch
                    label=f"Epoch {epoch}" if i == 0 else None,  # Add legend only once
                )

        plt.xticks(lr_positions + offsets.mean(), [f"lr={lr}" for lr in present_lrs])  # Use only present learning rates
        plt.ylabel('Average Score')
        plt.xlabel('Learning Rate')
        plt.title(f'Model Performance Comparison (Batch {batch})')
        plt.legend(title="Epochs")
        plt.tight_layout()
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        # Uncomment the line below to save the plot
        # plt.savefig(f'new_model_comparison_batch_{batch}.png')
        # print(f"Saved plot for Batch {batch} as 'new_model_comparison_batch_{batch}.png'")
        plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reprocess evaluation results and generate new plots grouped by batches')
    parser.add_argument('--pickle_file', type=str, default='evaluation_results_f.pkl', help='Path to the pickle file')
    args = parser.parse_args()

    reprocess_results(args.pickle_file)