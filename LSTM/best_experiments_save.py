# print_and_save_best_experiments.py
#summary_metrics_range2_all.txt
# Open and read file
#lstm__metrics_range3.txt


with open('lstm_metrics_all.txt', 'r') as f:
    lines = f.readlines()

experiments = []
current_exp = None
current_block = []
found_rmse = None

for line in lines:
    line = line.strip()

    if line.startswith('=== Experiment'):
        # If we have previous, store it
        if current_exp is not None and found_rmse is not None:
            experiments.append((current_exp, found_rmse, '\n'.join(current_block)))

        # Start new experiment
        try:
            current_exp = int(line.split()[2])
        except Exception:
            current_exp = None
        current_block = [line]
        found_rmse = None

    else:
        current_block.append(line)
        if line.startswith('rmse:'):
            try:
                rmse_value = float(line.split(':')[1].strip())
                found_rmse = rmse_value
            except Exception:
                found_rmse = None

# After finishing loop, save last experiment
if current_exp is not None and found_rmse is not None:
    experiments.append((current_exp, found_rmse, '\n'.join(current_block)))

# Sort by RMSE
experiments_sorted = sorted(experiments, key=lambda x: x[1])

# Print Top 5 best experiments
print("\nTop 5 experiments with smallest RMSE:\n")

for exp_id, rmse_val, full_text in experiments_sorted[:10]:
    print(f"=== Experiment {exp_id} === - RMSE: {rmse_val:.6f}\n")
    print(full_text)
    print("-" * 80)

# Save the Top 5 into a text file
with open('optimal_outputs.txt', 'w') as f_out:
    for exp_id, rmse_val, full_text in experiments_sorted[:5]:
        f_out.write(f"=== Experiment {exp_id} === - RMSE: {rmse_val:.6f}\n")
        f_out.write(full_text + '\n')
        f_out.write("-" * 80 + '\n')

print("\n Top 5 experiments were also saved into 'optimal_outputs.txt'.")
