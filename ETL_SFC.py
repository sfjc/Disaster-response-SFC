# import packages
import sys
import pandas as pd
import numpy as np
import random
import requests
import json

# ETL pipeline function

def etl_pipeline(path,messages="data/disaster_messages.csv",categories="data/disaster_categories.csv",table="data/dfall"):

    '''This loads the messages and processes them into a format suitable for passing to the Machine Learning pipeline'''

    # load messages dataset
    messages = pd.read_csv(messages)
    messages.head()

    # load categories dataset
    categories = pd.read_csv(categories)
    categories.head()

    # merge datasets
    df = pd.merge(categories, messages, how='inner', on = 'id')
    df.head()
    
    



    # create a dataframe of the 36 individual category columns

    seriescat = df.iloc[:,1]
    cats = seriescat.str.split(";",expand=True)
    cats.info()

    # select the first row of the categories dataframe
    row = cats.iloc[1,:]

    print(row)

    # extract a list of new column names for categories.


    category_colnames =  row.str[:-2]

    print(category_colnames)

    # rename the columns of `categories`
    cats.columns = category_colnames

    cats.head()

    cats.info()

    catsnum=cats
    for column in catsnum:
        # set each value to be the last character of the string
        catsnum[column] = cats[column].str[-1:]
    

        # convert column from string to numeric
        catsnum[column] = cats[column].astype(float)
    catsnum.head()
    catsnum.info()

    # drop the original categories column from `df`

    df = df.drop(columns=['categories'])

    df.head()
    df.info()

    # concatenate the original dataframe with the new `categories` dataframe



    dfall = pd.concat([df, catsnum], axis=1)

    dfall.info()

    # check number of duplicates


    dfall['duplicated'] = dfall.duplicated()


    alldups = dfall['duplicated'].sum()
    print(alldups)

    # drop duplicates

    dfall = dfall[dfall['duplicated'] != 1]


    # check number of duplicates
    alldups = dfall['duplicated'].sum()
    print(alldups)

    import sqlalchemy
    print("test")


    print(dfall)


    engine = sqlalchemy.create_engine('sqlite:///'+path) 

    dfall.to_sql(table, engine, index=False)




etl_pipeline('data/response.db')



