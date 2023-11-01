import pandas as pd
from clean_data import read_and_clean, determine_result

def test_determine_result():
    sample_data = {
        'Round': ['R1'],
        'Date': ['Tue Jan 01 2020'],
        'Team1': ['TeamA'],
        'FT': ['2–1'],
        'Team2': ['TeamB']
    }
    test_df = pd.DataFrame(sample_data)
    result = determine_result(test_df.iloc[0])
    assert result == 'H'

def test_read_and_clean_data():
    cleaned_data = read_and_clean('./csv/fixture_results.csv')
    assert 'Result' in cleaned_data.columns
