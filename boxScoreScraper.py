import requests as r
import bs4
import datetime
import pandas as pd

base_url = "https://basketball.realgm.com/international/scores/"
start_date = datetime.date(2019, 1, 1)
end_date = datetime.date(2019, 1, 8)
anchor_date = datetime.date(2100, 1, 1) # used in creating recency metric

def make_url(base_url, date):
    # takes a date object and returns the corresponding url
    y = str(date.year)
    m = str(date.month)
    if int(m) < 10: m = "0" + m
    d = str(date.day)
    if int(d) < 10: d = "0" + d
    return  base_url + y + "-" + m + "-" + d

def get_box_score_urls(base_url, date):
    # takes a date, grabs all "Box Score" links on the page
    url = make_url(base_url, date)
    page = r.get(url)
    soup = bs4.BeautifulSoup(page.text, features="lxml")
    link_elements = soup.findAll("a", string="Box Score")
    box_score_urls = ["https://basketball.realgm.com"+l.attrs['href'] for l in link_elements]
    return box_score_urls
    
def game_data_from_urls(urls):
    # takes a list of urls pointing to individual box score pages
    # returns a dataframe, with one row per url provided
    # df contains game/team level info, player level handled in player_data_from_urls
    df_rows = []
    for u in urls:
        print("accessing page: "+u+"\n")
        try:
            page = r.get(u)
        except Exception as e:
            print("\n Exception "+str(e))
        soup = bs4.BeautifulSoup(page.text, features="lxml")
        data = {}

        game_details = soup.find("div", class_="boxscore-gamedetails")
        away_home_teamnames = [a.text for a in game_details.h2.findAll("a")]
        data['away_team']  = away_home_teamnames[0]
        data['home_team'] = away_home_teamnames[1]
        league_gym_others = [p.text for p in game_details.findAll("p")]
        data['league_name'] = league_gym_others[1]
        data['gym_name'] = league_gym_others[2]

        game_summary = soup.find("div", class_="boxscore-gamesummary")
        tables = game_summary.findAll("tbody")
        # reading first table
        rows = tables[0].findAll("tr")
        away_row = rows[0]
        home_row = rows[1]
        away_row_unpacked = [d.text for d in away_row.findAll("td")]
        home_row_unpacked = [d.text for d in home_row.findAll("td")]
        data['away_record'] = away_row_unpacked[0]
        data['away_q1'] = away_row_unpacked[1]
        data['away_q2'] = away_row_unpacked[2]
        data['away_q3'] = away_row_unpacked[3]
        data['away_q4'] = away_row_unpacked[4]
        data['away_final'] = away_row_unpacked[5]
        data['home_record'] = home_row_unpacked[0]
        data['home_q1'] = home_row_unpacked[1]
        data['home_q2'] = home_row_unpacked[2]
        data['home_q3'] = home_row_unpacked[3]
        data['home_q4'] = home_row_unpacked[4]
        data['home_final'] = home_row_unpacked[5]
        # reading second table, 'Advanced'
        rows = tables[1].findAll("tr")
        away_row = rows[0]
        home_row = rows[1]
        away_row_unpacked = [d.text for d in away_row.findAll("td")]
        home_row_unpacked = [d.text for d in home_row.findAll("td")]
        data['away_poss'] = away_row_unpacked[1]
        data['away_ORtg'] = away_row_unpacked[2]
        data['away_DRtg'] = away_row_unpacked[3]
        data['home_poss'] = home_row_unpacked[1]
        data['home_ORtg'] = home_row_unpacked[2]
        data['home_DRtg'] = home_row_unpacked[3]
        # reading third table, 'Four Factors'
        rows = tables[2].findAll("tr")
        away_row = rows[0]
        home_row = rows[1]
        away_row_unpacked = [d.text for d in away_row.findAll("td")]
        home_row_unpacked = [d.text for d in home_row.findAll("td")]
        data['away_eFGper'] = away_row_unpacked[1]
        data['away_TOper'] = away_row_unpacked[2]
        data['away_ORper'] = away_row_unpacked[3]
        data['away_FTR'] = away_row_unpacked[4]
        data['home_eFGper'] = home_row_unpacked[1]
        data['home_TOper'] = home_row_unpacked[2]
        data['home_ORper'] = home_row_unpacked[3]
        data['home_FTR'] = home_row_unpacked[4]

        df_rows.append(data)

    df = pd.DataFrame.from_dict(df_rows)

    return df

def player_data_from_urls(urls):
    # takes a list of urls pointing to box score pages
    # returns a df with indiviudal player stats
    # will implement later, I suspect we already have enough data
    # (in terms of number of features, that is)
    pass

def date_str(date, c):
    # takes a date, returns a string MMcDDcYYYY
    d = str(date.day)
    m = str(date.month)
    y = str(date.year)
    return m+c+d+c+y
    
def df_from_date(base_url, date):
    # takes a base_url, date, returns a df with data about
    # every game played on that day
    # the purpose of this function is to add columns to the
    # df from game_data_from_urls with info about the date
    urls = get_box_score_urls(base_url, date)
    df = game_data_from_urls(urls)

    df['date_played'] = date_str(date, "/")

    delta = anchor_date - date
    df['staleness'] = delta.days

    return df

def df_from_range(base_url, start_date, end_date):
    # makes a big df with data about every game played bw
    # the given dates, inclusive
    dfs = []
    delta = end_date - start_date
    for i in range(delta.days + 1):
        current_date = start_date + datetime.timedelta(days=i)
        df = df_from_date(base_url, current_date)
        dfs.append(df)
    return pd.concat(dfs)

def to_csv(base_url, start_date, end_date, path=""):
    # gets data about games in range, exports to csv
    filename = "box_scores_" + date_str(start_date, "_") + "_" + date_str(end_date, "_") + ".csv"
    df = df_from_range(base_url, start_date, end_date)
    df.to_csv(path+filename)

if __name__ == "__main__":
    to_csv(base_url, start_date, end_date)