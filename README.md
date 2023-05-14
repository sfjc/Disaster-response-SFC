
# Disaster response ML pipeline project
by S John Cody


## Dataset and motivation

This analyzes disaster data from Appen to build a model for an API that classifies disaster messages.

## Files

ETL_SFC.py 

This:
    Loads the messages and categories datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

listings.csv (3818 entries, describes individual properties)

reviews.csv (84849 entries, gives renter reviews)

## Summary of Findings

We see in this data, among other things, that the weekly cyclical variations in airbnb bookings diminish through the year 2016, that there are 
two huge drops in availability that may be related to significant events, and that overall availability increases towards the end of the year. There 
is a (very weak) positive association between cancellation policy and average price, and we find the ten words in property summaries most associated 
with higher prices.

## Acknowledgements

Would like to acknowledge airbnb for making this data available.

