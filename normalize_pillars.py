import pandas as pd
import os

# Define file paths
input_file = 'un_report_api/app/required_csvs/annual_scores.csv'
output_dir = 'normalized_data'
output_file = os.path.join(output_dir, 'normalized_pillar_scores.csv')

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

try:
    # Read the data
    df = pd.read_csv(input_file)

    # Identify pillar score columns
    pillar_cols = ['Pillar 1 Score', 'Pillar 2 Score', 'Pillar 3 Score']

    # Min-Max Normalization
    for col in pillar_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        # Handle case where max and min are the same to avoid division by zero
        if max_val - min_val > 0:
            df[f'{col} Normalized'] = 100 * (df[col] - min_val) / (max_val - min_val)
        else:
            df[f'{col} Normalized'] = 0  # Or 50, or some other constant, depending on desired behavior

    # Select columns for the new CSV
    output_cols = ['Country name', 'Year'] + [f'{col} Normalized' for col in pillar_cols]
    normalized_df = df[output_cols]

    # Save the normalized data
    normalized_df.to_csv(output_file, index=False)

    print(f"Successfully normalized pillar scores and saved to '{output_file}'")

except FileNotFoundError:
    print(f"Error: The file '{input_file}' was not found.")
except Exception as e:
    print(f"An error occurred: {e}") 