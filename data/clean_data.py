"""
Cleans data that is read from a csv
"""
import pandas as pd

def read_and_clean(file_name='./csv/fixture_results.csv'):
    """
    Reads data from csv and cleans it
    """
    # Read the CSV file (assuming it's named 'matches.csv')
    fixture_df = pd.read_csv(file_name, sep=',')

    # Rename columns for clarity
    fixture_df.columns = ['Round', 'Date', 'Team1', 'FT', 'Team2']

    # Replace the special characters with a hyphen (if needed)
    fixture_df['FT'] = fixture_df['FT'].str.replace('‚Äì', '-')

    # Convert the 'Date' column to a proper datetime format
    fixture_df['Date'] = pd.to_datetime(fixture_df['Date'], format='%a %b %d %Y')

    # Apply the function to create a new 'Result' column
    fixture_df['Result'] = fixture_df.apply(determine_result, axis=1)

    return fixture_df

# Define a function to determine the result based on the scores
def determine_result(row):
    """
    Determines the result as H, A or D
    """
    team1_score, team2_score = map(int, row['FT'].replace('–', '-').split('-'))
    if team1_score > team2_score:
        return 'H'  # Home Win
    if team1_score < team2_score:
        return 'A'  # Away Win
    return 'D'  # Draw


if __name__ == "__main__":
    cleaned_data = read_and_clean()

    # Save the DataFrame to a CSV file
    cleaned_data.to_csv('./csv/cleaned_fixture_results.csv', index=False)

    # Print a message to confirm the save
    print("DataFrame saved to 'cleaned_fixture_results.csv'")
