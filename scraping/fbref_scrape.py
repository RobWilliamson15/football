"""
Scrape fbref for premier league fixtures and results
"""
import csv
from datetime import datetime
import requests
from bs4 import BeautifulSoup

# Define a list of seasons to loop through
# FBRef limit of 20 calls
seasons = range(2020, 2023)

# Initialize an empty list to store all data
all_data = []

for season in seasons:

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

        # Append the data for this season to the overall data list
        all_data.extend(main_list)

with open('./data/fixture_results.csv', 'w', newline='', encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Round", "Date", "Team 1", "FT", "Team 2"])
    for row in all_data:
        writer.writerow(row.split(','))
