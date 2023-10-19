import pandas as pd

# Read the CSV file (assuming it's named 'matches.csv')
df = pd.read_csv('fixture_results.csv', sep=',')

# Rename columns for clarity
df.columns = ['Round', 'Date', 'Team1', 'FT', 'Team2']

# Replace the special characters with a hyphen (if needed)
df['FT'] = df['FT'].str.replace('‚Äì', '-')

# Convert the 'Date' column to a proper datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%a %b %d %Y')

# Define a function to determine the result based on the scores
def determine_result(row):
    team1_score, team2_score = map(int, row['FT'].replace('–', '-').split('-'))
    if team1_score > team2_score:
        return 'H'  # Home Win
    elif team1_score < team2_score:
        return 'A'  # Away Win
    else:
        return 'D'  # Draw

# Apply the function to create a new 'Result' column
df['Result'] = df.apply(determine_result, axis=1)

# Save the DataFrame to a CSV file
df.to_csv('cleaned_fixture_results.csv', index=False)

# Print a message to confirm the save
print("DataFrame saved to 'cleaned_fixture_results.csv'")
