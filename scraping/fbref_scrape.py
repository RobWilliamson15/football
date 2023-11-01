"""
Scrape fbref for premier league fixtures and results
"""
import csv
from datetime import datetime
import requests
from bs4 import BeautifulSoup

def scrape_fixtures_for_season(season):
    # Get season start and end
    season_start = season
    season_end = season + 1

    # Create standings_url
    standings_url =f"https://fbref.com/en/comps/9/{season_start}-{season_end}/schedule/{season_start}-{season_end}-Premier-League-Scores-and-Fixtures"
    # Get data
    data = requests.get(standings_url)
    soup = BeautifulSoup(data.text, features='lxml')
    standings = soup.select('table.stats_table')[0]
    
    main_list=[]
    for team in standings.find_all('tbody'):
        rows = team.find_all('tr')
        for i in rows:
            # round and day
            round = i.find('th',{'data-stat':'gameweek'})
            if round is not None:
                Round = round.text
            day = i.find('td',{'data-stat':'dayofweek'})
            if day is not None:
                Day=day.text


            #date
            date=i.find('td',{'data-stat':'date'})
            date = date.find('a')
            if date is not None:
                Date=date.text
                #converting date formats
                date_obj = datetime.strptime(Date, "%Y-%m-%d")
                Date = date_obj.strftime("%b %d %Y")


            #home_team
            ht= i.find('td',{'data-stat':'home_team'})
            ht=ht.find('a')
            if ht is not None:
                Ht=ht.text

            #score
            sc= i.find('td',{'data-stat':'score'})
            sc=sc.find('a')
            if sc is not None:
                Sc=sc.text

            #away_team
            at= i.find('td',{'data-stat':'away_team'})
            at=at.find('a')
            if at is not None:
                At=at.text

            #each game
            if Round =='':
                continue
            game = Round + ',' + Day + ' ' + Date + ',' +Ht + ',' + Sc + ',' + At
            main_list.append(game)
        main_list.sort(key = lambda x: int(x.split(',')[0]))

    return main_list

def save_to_csv(data_list, filename):
    with open(filename, 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Date", "Team 1", "FT", "Team 2"])
        for row in data_list:
            writer.writerow(row.split(','))

if __name__ == "__main__":
    seasons = range(2020, 2023)
    all_data = []
    for season in seasons:
        all_data.extend(scrape_fixtures_for_season(season))
    save_to_csv(all_data, '../data/csv/fixture_results.csv')
