import pandas as pd
import argparse

# methods to create new columns
def add_home_minus_away(df):
    # adds a column to the df with the home team's final score - away team's final score
    # represents the opposite of the minimum spread one could have bet on and won
    df['home_minus_away'] = df['home_final'] - df['away_final']

def add_point_total(df):
    # adds a column with the total points scored
    df['total'] = df['home_final'] + df['away_final']


if __name__ == "__main__":
    # grabbing command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help='.csv file to process')
    parser.add_argument('--out', type=str, help='output file name, if none provided, overwrites target')

    args = parser.parse_args()
    target = args.target
    out = target if not args.out else args.output

    # importing data
    df = pd.read_csv(target, index_col=0)
    df.reset_index()

    # editing df
    add_home_minus_away(df)
    add_point_total(df)
    df.to_csv(out)