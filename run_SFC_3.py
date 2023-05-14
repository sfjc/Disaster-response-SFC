import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import *
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):

    '''This creates tokens from the data'''
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/response.db')
df = pd.read_sql_table("data/dfall", engine)



# load model
model = joblib.load("../classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    df2=df.drop(df.columns[[0,1,2,3,4]], axis = 1)

    df3=df2.drop(columns=["duplicated"], axis = 1)

    dfchart3 = df3.apply(pd.value_counts)


    dfchart3 = dfchart3.iloc[1]

#    dfchart3.info()

    dfchart3=dfchart3.sort_values(ascending=False)

    dfchart3frame = dfchart3.to_frame()

    dfchart3frame.info()

    dfchart3frame.index = dfchart3frame.index.str.replace("_", " ")

    dfchart3frame.index = dfchart3frame.index.str.capitalize()

    dfchart3frame=dfchart3frame.fillna(0)

    dfchart3frameser = dfchart3frame.iloc[:,0]

#    dfchart3frameser=dfchart3frameser.reset_index()

#    dfchart3frameser=dfchart3frameser.drop(columns=["index"], axis = 1)

#    dfchart3frameser = dfchart3frameser.iloc[:,0]


    x2= list(dfchart3frameser.index)

    y2 = dfchart3frameser

    dfallhm=df2

#    dfallhm['genre']=dfallhm['genre'].astype('category').cat.codes

    dfallhm=dfallhm.drop(columns=["duplicated"], axis = 1)

    dfallhm=dfallhm.drop(columns=['child_alone'], axis = 1)

#    dfallhm=dfallhm.drop(columns=['genre'])

    dfallhm.columns = dfallhm.columns.str.replace("_", " ")

    dfallhm.columns = dfallhm.columns.str.capitalize()

    dfallhm_corr = dfallhm.corr()

    x3 = dfallhm_corr.columns

    y3 = dfallhm_corr.index

    z3 = np.array(dfallhm_corr)

    text3=dfallhm_corr.values


    
    # create visuals
    # 3 visuals:
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=x2,
                    y=y2
                )
            ],

            'layout': {
                'title': 'Number of messages',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Content"
                }
            }
        },
        {
            'data': [
                Heatmap(
                    x = x3,
                    y = y3,
                    z = z3,
                    text=text3
                )
            ],

            'layout': {
                'title': 'Heatmap - correlation of factors',
                'yaxis': {
                    'title': "Factors"
                },
                'xaxis': {
                    'title': "Factors"
                }
            }
        }
    ]
        # create visuals

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()
