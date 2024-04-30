import re
from collections import defaultdict

def calculate_latent_averages(file_path):
    # Dictionary to store cumulative sums and counts for each latent dimension
    latent_stats = defaultdict(lambda: {'mean_sum': 0, 'log_var_sum': 0, 'count': 0})

    # Regular expression patterns to extract mean and log var values
    mean_pattern = r"Mean:\s*([-+]?\d*\.\d+|\d+)"
    log_var_pattern = r"Log Var:\s*([-+]?\d*\.\d+|\d+)"

    # Read data from file and calculate cumulative sums for each latent dimension
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("Training with latent size:"):
                latent_size = int(line.split()[-1])
            if "Mean: " in line:
                mean_match = re.search(mean_pattern, line)
                if mean_match:
                    mean = float(mean_match.group(1))
            if "Log Var: " in line:
                log_var_match = re.search(log_var_pattern, line)
                if log_var_match:
                    log_var = float(log_var_match.group(1))
                    # Update cumulative sums for the current latent dimension
                    latent_stats[latent_size]['mean_sum'] += mean
                    latent_stats[latent_size]['log_var_sum'] += log_var
                    latent_stats[latent_size]['count'] += 1
                else:
                    print('kat gaya')

    # Calculate averages for each latent dimension
    latent_averages = {}
    for latent_size, stats in latent_stats.items():
        mean_avg = stats['mean_sum'] / stats['count']
        log_var_avg = stats['log_var_sum'] / stats['count']
        latent_averages[latent_size] = {'Mean': mean_avg, 'Log Var': log_var_avg}
    
    return latent_averages

def write_latent_averages(latent_averages, output_file):
    # Write averages to a new file
    with open(output_file, 'w') as file:
        for latent_size, averages in latent_averages.items():
            file.write(f"Latent Size: {latent_size}, Mean: {averages['Mean']}, Log Var: {averages['Log Var']}\n")

if __name__ == "__main__":
    input_file = "out.log"
    output_file = "latent_averages.txt"
    averages = calculate_latent_averages(input_file)
    write_latent_averages(averages, output_file)