#!/usr/bin/env python
# coding: utf-8


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# fetching all the data from all the csv files and converting to them to numpy arrays
df_2017 = pd.read_csv('./2017.csv').to_numpy()
df_2018 = pd.read_csv('./2018.csv').to_numpy()
df_2019 = pd.read_csv('./2019.csv').to_numpy()


# Making a list of all the countries in the dataset according to 2017 data
countries_2017 = []
for i in range(len(df_2017)):
    countries_2017.append((df_2017[i][1]).upper().lower())
countries_2018 = []
for i in range(len(df_2017)):
    countries_2018.append((df_2018[i][1]).upper().lower())
countries_2019 = []
for i in range(len(df_2017)):
    countries_2019.append((df_2019[i][1]).upper().lower())


def avg_gdp(arr):
    gdp = [row[3] for row in arr]
    return np.mean(gdp)


def max_gdp(arr):
    gdp = [row[3] for row in arr]
    return np.max(gdp)


def min_gdp(arr):
    gdp = [row[3] for row in arr]
    return np.min(gdp)


def gdp_graph(country):
    print("GDP per capita")
    print("For year 2017 is", df_2017[countries_2017.index(country)][3])
    print("The average is", avg_gdp(df_2017))
    print("The maximum is", max_gdp(df_2017))
    print("The minimum is", min_gdp(df_2017))
    print()

    print("For year 2018 is", df_2018[countries_2018.index(country)][3])
    print("The average is", avg_gdp(df_2018))
    print("The maximum is", max_gdp(df_2018))
    print("The minimum is", min_gdp(df_2018))
    print()

    print("For year 2019 is", df_2019[countries_2019.index(country)][3])
    print("The average is", avg_gdp(df_2019))
    print("The maximum is", max_gdp(df_2018))
    print("The minimum is", min_gdp(df_2019))
    print()

    gdps = [df_2017[countries_2017.index(country)][3], df_2018[countries_2018.index(
        country)][3], df_2019[countries_2019.index(country)][3]]

    years = [2017, 2018, 2019]
    # gdps = [df_2017[countries_2017.index(country)][3],df_2018[countries_2018.index(country)][3],df_2019[countries_2019.index(country)][3]]
    avg_gdps = [avg_gdp(df_2017), avg_gdp(df_2018), avg_gdp(df_2019)]

    X_axis = np.arange(len(years))
    plt.bar(X_axis - 0.2, gdps, 0.4, label=country)
    plt.bar(X_axis + 0.2, avg_gdps, 0.4, label='Average', tick_label=years)
    plt.xlabel("Years")
    plt.ylabel("GDP per capita")
    plt.title("GDP per capita")
    plt.legend()
    plt.show()


def avg_health(arr):
    health = [row[5] for row in arr]
    return np.mean(health)


def max_health(arr):
    health = [row[5] for row in arr]
    return np.max(health)


def min_health(arr):
    health = [row[5] for row in arr]
    return np.min(health)


def health_graph(country):
    print("Healthy life expectancy")
    print("For year 2017 is", df_2017[countries_2017.index(country)][5])
    print("The average is", avg_health(df_2017))
    print("The maximum is", max_health(df_2017))
    print("The minimum is", min_health(df_2017))

    print()
    print("For year 2018 is", df_2018[countries_2018.index(country)][5])
    print("The average is", avg_health(df_2018))
    print("The maximum is", max_health(df_2018))
    print("The minimum is", min_health(df_2018))

    print()
    print("For year 2019 is", df_2019[countries_2019.index(country)][5])
    print("The average is", avg_health(df_2019))
    print("The maximum is", max_health(df_2019))
    print("The minimum is", min_health(df_2019))

    print()

    healths = [df_2017[countries_2017.index(country)][3], df_2018[countries_2018.index(
        country)][3], df_2019[countries_2019.index(country)][3]]

    years = [2017, 2018, 2019]
    # gdps = [df_2017[countries_2017.index(country)][3],df_2018[countries_2018.index(country)][3],df_2019[countries_2019.index(country)][3]]
    avg_healths = [avg_health(df_2017), avg_health(
        df_2018), avg_health(df_2019)]

    X_axis = np.arange(len(years))
    plt.bar(X_axis - 0.2, healths, 0.4, label=country)
    plt.bar(X_axis + 0.2, avg_healths, 0.4, label='Average', tick_label=years)
    plt.xlabel("Years")
    plt.ylabel("Healthy life expectancy")
    plt.title("Healthy life expectancy")
    plt.legend()
    plt.show()


country = ""
while(country not in countries_2017):
    print("Please enter a valid country (all in lower case)")
    country = input()


print("The following is the data of", country)
print("The Overall Rank")
print("For year 2017 is", df_2017[countries_2017.index(country)][0])
print("For year 2018 is", df_2018[countries_2018.index(country)][0])
print("For year 2019 is", df_2019[countries_2019.index(country)][0])
ranks = [df_2017[countries_2017.index(country)][0], df_2018[countries_2018.index(
    country)][0], df_2019[countries_2019.index(country)][0]]
if(ranks[0] > ranks[2]):
    if(ranks[0]-ranks[2] <= 3):
        print('As we can see the overall rank is almost the same')
    else:
        print('As we can see the overall rank has been improved')
elif(ranks[0] < ranks[2]):
    if(ranks[2]-ranks[0] <= 3):
        print('As we can see the overall rank is almost the same')
    else:
        print('As we can see the overall rank is getting worse')
else:
    print('As we can see the overall rank is the same')


print("Enter 1 to see stats of gdp")
print("Enter 2 to see stats of healthy life expectancy")

choice = int(input())


if(choice == 1):
    gdp_graph(country)
elif(choice == 2):
    health_graph(country)
