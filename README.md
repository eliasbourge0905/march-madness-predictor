# March Madness Prediction
## Project Overview
This project aims to take historical data from NCAA tourtments and predict teams wins and losses, I developed and compared logistical regression and Random Forest classifiers against a simple seed baseline. The goal is to determine whether team performance statistics can improve tournament predictions beyond simply selecting the higher-seeded team.

## Installation and Setup
### Codes and Resources Used
- **Editor:** JupyterLab
- **Language:** Python
- **Environment:** Jupyter Notebook
- 
### Python Packages Used
- **Data Manipulation:** pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** scikit-learn

## data
### source data
cbb.csv 
https://www.kaggle.com/datasets/andrewsundberg/college-basketball-dataset 
This is the most foundtational dataset for this porgect it incuded all of the basic statistics for every team, like fg%. wins/losses, and seed data

The next 3 data sets are from this march maddness data kaggle 
https://www.kaggle.com/datasets/nishaanamin/march-madness-data/data

538 Ratings.csv
This is a power rating for each team from FiveThirtyEight, it is a metric that mesures a teams sucsess on a nutrual court

AP poll data.csv
This is a dataset that contains weekly ap poll data for each team that was ranked on the ap poll

KenPom Barttorvik.csv
These are more advanced metrics from both Kenpom and Barttorvik, these include metrics such as tempo and offenvise and deffesive efficency.

Last I used two data sets which are the win loss data that I needed to predict the wins and losses, they came from a google march maddsess compition from last march

https://www.kaggle.com/competitions/march-machine-learning-mania-2025/data?utm_source=chatgpt.com 

From this kaggle I used MTeams.csv which just gave the teams are thier team id and MNCAATourneyCompactResults.csv which incuded the team id and the win loss data for each game

### Data Preprocessing
For all the datasets I used I had to clean and then merge them into one csv file that was ready for modeling
The first thing that I did was I did a basic check on all data sets and made sure they would be good for the spesific models I wanted, whule doing that I noticed that AP poll might be hard to model since its giving weekly stats so I made it so each spesific team just had the most recent ap poll from that year. The next thing I did was standardised the years for all my data sources, they all had different starting and ending years, so it limited the data to 1016-2024 skipping 2020 since that was the year with no tourtment. The next I had to do was standardize the team names which incuded going through all data and mapping out any differentces for example, MIchigan state, and Michigan St, after getting that sorted I Merged data sets and did a few checks to make sure the data was merged correctly.

The next thing I had to do Data Preprocessing wise was matching the win loss data to the previous data set that I mentioned, first thing I had to do was clean the win loss data and metge them for that I had to find the equevilent id from Mteams.csv to MNCAATourneyCompactResults after that I had standardize the names between the new merged win loss data and the csv containing all my featues. I used a similar process as the first time did that, where I used mapping to change and name discrepicies. Then I creted an A and B team so that the model could more acruatly predict the wins and losses from each game, lasttly I merged this win loss data with the csv with all other data.

### Data Leakage
One thing I ran into while cleaning the data was that win percentage incuded the wins and lossed from tourtment games, this could have a fairly large effect on win percentage adding up to 7 games, so I decided to take win percentage out of the data set,win percentage something that I think would make my data a lot stronger so in the future I would like to add win percentage data from before tourtments.

## Feature Engineering
The big thing I did for feature engneering was for all features I wanted I created a difference varible for them and used that for my models, it was just subtracing the team A statistic and team B statistic and getting the difference from there, this would make it easier for the models to compare two teams matchups.

##  Models
I used two models in this project Logicial regression and Random forres classifier, and compared the data that I got from both to just picking the team witht he higher seed to win every game

I chose Logicial regression because it is generly a good model for making binary decisions like yes or no, and since I just want to so say if one team would beat another I thought this would be a strong model to use

Random Forest I chose becuase it is a nonlinear tree based classifier so I thought that it could make different connections in the data maybe leading to better results, it also didn't require the same scaling that the Logicial regression model used

## Results and evaluation

| Model | Accuracy | ROC-AUC | Log Loss | Brier Score |
|---|---:|---:|---:|---:|
| Seed Baseline | 65.67% | — | — | — |
| Logistic Regression | 68.66% | 0.780 | 0.570 | 0.198 |
| Random Forest | 68.66% | 0.805 | 0.548 | 0.184 |








