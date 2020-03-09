import pandas as pd
import argparse

csvs = ["decade/box_scores_1_1_2010_1_1_2011.csv", 
"decade/box_scores_1_1_2011_1_1_2012.csv", 
"decade/box_scores_1_1_2012_1_1_2013.csv", 
"decade/box_scores_1_1_2013_1_1_2014.csv", 
"decade/box_scores_1_1_2014_1_1_2015.csv", 
"decade/box_scores_1_1_2015_1_1_2016.csv", 
"decade/box_scores_1_1_2016_1_1_2017.csv", 
"decade/box_scores_1_1_2017_1_1_2018.csv", 
"decade/box_scores_1_1_2018_1_1_2019.csv",
]


if __name__ == "__main__":
    # grabbing command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=str, help='output file name')

    args = parser.parse_args()
    out = args.out

    #df = pd.read_csv(target, index_col=0)
    #df.reset_index()
    dfs = [pd.read_csv(f, index_col=0) for f in csvs]
    for df in dfs:
        df.reset_index()
    df = pd.concat(dfs)

    df.to_csv(out)