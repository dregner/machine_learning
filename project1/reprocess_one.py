import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the saved results from the pickle file
def load_results(pickle_file):
    with open(pickle_file, 'rb') as f:
        results = pickle.load(f)
    return results

# Reprocess results and generate a single plot for all models
def reprocess_all_models(pickle_file, lr_filter=['5e-05', '00001', '00005', '00025', '0005', '001']):
    results = load_results(pickle_file)

    # Group results by model
    grouped_results = {}
    for mf, (mean, std) in results.items():
        if 'car_cnn_model_' in mf:
            label = mf.split('car_cnn_model_')[-1].split('.')[0]  # Extract model name

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

            # Group by model name
            if label not in grouped_results:
                grouped_results[label] = []
            grouped_results[label].append((epoch, batch, lr, mean, std))  # Store epoch, batch, lr, mean, std
        else:
            print(f"Warning: Skipping file {mf} as it does not contain '_epoch' or '_batch'")

    # Generate a single plot for all models
    plt.figure(figsize=(14, 8))

    # Sort models by name and epochs
    sorted_models = sorted(grouped_results.items(), key=lambda x: x[0])  # Sort by model name
    bar_width = 0.2  # Width of each bar
    x_positions = np.arange(len(sorted_models))  # X positions for each model

    for i, (model_name, data) in enumerate(sorted_models):
        # Sort data by epoch
        data = sorted(data, key=lambda x: x[0])  # Sort by epoch
        means = [d[3] for d in data]  # Extract mean scores
        stds = [d[4] for d in data]  # Extract standard deviations
        epochs = [d[0] for d in data]  # Extract epochs

        # Plot bars for each model
        plt.bar(
            x_positions[i] + np.arange(len(epochs)) * bar_width,
            means,
            yerr=stds,
            capsize=5,
            width=bar_width,
            label=f"{model_name} (Epochs: {', '.join(map(str, epochs))})"
        )

    # Set x-axis labels
    plt.xticks(x_positions + (len(epochs) - 1) * bar_width / 2, [model[0] for model in sorted_models], rotation=45, ha='right')
    plt.ylabel('Average Score')
    plt.xlabel('Model Name')
    plt.title('Model Performance Comparison (All Models)')
    plt.legend(title="Models and Epochs")
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # Uncomment the line below to save the plot
    # plt.savefig('all_models_comparison.png')
    # print("Saved plot as 'all_models_comparison.png'")
    plt.show()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Reprocess evaluation results and generate a single plot for all models')
    parser.add_argument('--pickle_file', type=str, default='evaluation_results_noseed.pkl', help='Path to the pickle file')
    args = parser.parse_args()

    reprocess_all_models(args.pickle_file)