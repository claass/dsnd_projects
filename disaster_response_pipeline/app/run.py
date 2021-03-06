import json
import plotly
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)


def tokenize(text):
    """
    Creates tokes from a given text corpus. Includes replacing
    urls and @mentions with respective placeholders, stripping out special
    characters, lemmatizing, and stripping

    Args:
        text (str): text input as string

    Returns:
        list of tokens
    """

    # Replace URLS with placeholder
    url_regex = "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    # Replace mentions with placeholder
    mention_regex = "/^(?!.*\bRT\b)(?:.+\s)?@\w+/i"
    detected_mentions = re.findall(mention_regex, text)
    for mention in detected_mentions:
        text = text.replace(mention, "mentionplaceholder")

    # Tokenize and lemmatize
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# load data
engine = create_engine('sqlite:///../data/DisasterRelief.db')
df = pd.read_sql_table('Tweets', engine)

# load model
model = joblib.load("../models/model.pickle")


# index webpage displays visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    """
    Handler for the index page. Includes simple calculations and passes
    data for summary graphs to the frontend.
    """

    # extract data for barchart
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # extract data for piechart
    sub = df.drop(['message', 'genre', 'id'], axis=1)
    sum = sub.sum(axis=0)
    sum.sort_values(ascending=False, inplace=True)
    class_labels = sum.index.values
    class_counts = sum.values
    # print(counts, file=sys.stderr)

    # create visuals
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
                    x=class_labels,
                    y=class_counts
                )
            ],

            'layout': {
                'title': 'Count of Training Examples by Class',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                },
                'height': 300,
            }
        },
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    """
    Handler for classification task submission.
    """
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
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
