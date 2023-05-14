
# Disaster response ML pipeline project
by S John Cody


## Dataset and motivation

This project analyzes disaster data from Appen to build a model for an API that classifies disaster messages.

## Files

ETL_SFC.py 

This Python script:

    Loads the messages and categories datasets
    Merges the two datasets
    Cleans the data
    Stores it in a SQLite database

ML_SFC_3.py

This Python script is a machine learning pipeline that:

    Loads data from the SQLite database
    Splits the dataset into training and test sets
    Builds a text processing and machine learning pipeline
    Trains and tunes a model using GridSearchCV
    Outputs results on the test set
    Exports the final model as a pickle file
    
run_SFC_3.py

    This is a web app where an emergency worker can input a new message and get 
    classification results in several categories.
    The web app displays visualizations of the data.

## IDE process:
    
### Instructions:
    
1. Run the following commands in the project's root directory

    - To run ETL pipeline that cleans data and stores in database
        `python data/ETL_SFC.py data/disaster_messages.csv data/disaster_categories.csv data/response.db`
    - To run ML pipeline that trains classifier and saves
        `python models/ML_SFC_3.py data/response.db models/classifier.pkl`

2. Go to `app` directory: `cd app`

3. Run the web app: `run_SFC_3.py`

4. Click the `PREVIEW` button to open the homepage







## Acknowledgements

Would like to acknowledge Appen for making this useful data available.


