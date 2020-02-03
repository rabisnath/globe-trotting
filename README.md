# globe-trotting

## About

This project includes a web scraper to retrieve box scores of international basketball games from https://basketball.realgm.com/international/, and tools to train machine learning models to predict game outcomes.

## Install

1. Clone the repository
2. Create a virtual environment (if you want) `virtualenv my_env` `source /my_env/bin/activate`
3. `pip install -r requirements.txt`

## Tools

### boxScoreScraper.py

Running boxScoreScraper from the command line will create a .csv in the working directory with stats about every basketball game played between 'start-date' and 'end-date'. As of right now, these dates are hardcoded in at the top of the script.

### addColumns.py

This script has a positional argument, `target`, which takes a .csv. The script adds two new columns to the dataset, one with the difference between the home and away team final scores, and another with the total points scored. If the optional argument `--out` is specified, the new dataset will be saved to that file, instead of overwriting the target file.

### pointTotals.py

Trains a regression model to predict the total points score in a given game. Takes a positional argument `target` which should be a .csv containing the training data.

This is currently incomplete. Right now, it trains on an arbitrary set of features that don't reflect real world conditions, and no thought has been put into various ways to reshape the data before training.
