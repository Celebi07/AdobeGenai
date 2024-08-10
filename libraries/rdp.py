import pandas as pd
from rdp import rdp

def straightenLine(lines) : 
    # Define the input and output CSV file paths
    input_csv = 'problems/frag2.csv'

    # Read the CSV file into a DataFrame
    df = pd.read_csv(input_csv, header=None)

    # Create an empty DataFrame for the output
    output_df = pd.DataFrame(columns=[0, 1, 2, 3])

    # Iterate over unique path numbers
    for path_number in df[0].unique():
        # Filter rows for the current path number
        path_df = df[df[0] == path_number]
    
        # Extract x and y coordinates
        coordinates = path_df[[2, 3]].values
    
        # Apply RDP algorithm to simplify the path
        simplified_coordinates = rdp(coordinates, epsilon=10.51)  # Adjust epsilon as needed
    
        # Create a DataFrame for the simplified path
        simplified_df = pd.DataFrame(simplified_coordinates, columns=[2, 3])
        simplified_df[0] = path_number
        simplified_df[1] = 0  # Add a dummy column for consistency with input format
    
        # Append to output DataFrame
        output_df = pd.concat([output_df, simplified_df], ignore_index=True)

        return output_df