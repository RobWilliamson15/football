# Football Data Analysis

This repository contains Python code snippets for various football-related tasks, including data scraping, data manipulation, and model training.

## Model Training and Testing

### Description

test.py trains and tests a machine learning model based on scraped football data from fbref.com. The model predicts the outcomes of football matches (Win, Draw, or Lose) based on the teams playing.

### Usage

- Ensure you have the required libraries installed (e.g., PyTorch, pandas).
- Run the code using `python test.py`.
- The code will preprocess the data, train a neural network model, and evaluate its performance.

## Scraping Premier League Fixtures and Results

### Description

fbref_scrape.py scrapes Premier League fixtures and results data from fbref.com and saves it in a CSV file. It includes season-specific data retrieval and data parsing.

### Usage

- Install the required libraries (requests, BeautifulSoup).
- Run the script using `python scraping/fbref_scrape.py`.
- The data will be scraped and saved in a CSV file named `data/fixture_results.csv`.

## Cleaning Fixture Results

### Description

clean_data.py reads a CSV file containing Premier League fixture results, cleans the data, and saves the cleaned data to a new CSV file. It includes column renaming, special character replacement, date formatting, and match result determination.

### Usage

- Ensure you have pandas installed.
- Run the code using `python data/clean_data.py`.
- The script will read, clean, and save the data as `cleaned_fixture_results.csv`.

## Requirements

The code snippets have specific library requirements, which are mentioned in each snippet's description. You can install the required libraries using `pip install <library-name>` or by running `pip install -r requirements.txt`
