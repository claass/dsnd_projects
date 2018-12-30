# import libraries
import sys
import re
import nltk
import pickle
import pandas as pd
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.multioutput import MultiOutputClassifier

# Downloads potentially neccesary for the tokenization process
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)


def load_data(database_filepath):
    """
    Loads the training data from the given sqlite database.

    Args:
        database_filepath (str): filepath to the sqlite database containing
            the training data

    Returns:
        X (pd.DataFrame): Features for training
        y (pd.DataFrame): Target values for training (36 different classes)
        categories (list): List of column names for easy access
    """

    # Connect to database and get all tweets
    engine = create_engine('sqlite:///'+database_filepath)
    result = engine.execute("SELECT * FROM Tweets")
    df = pd.DataFrame(data=result.fetchall(), columns=result.keys())

    # Splitting data into target and feautres
    X = df.message.values
    y = df.drop(['id', 'message', 'original', 'genre'], axis=1)
    categories = y.columns.values

    return X, y, categories


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


def build_model():
    """
    Compiles the ML pipeline. For simplicity the parameter space
    for GridSearchCV is kept at a minimum.

    Returns:
        GridSearchCV object
    """

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    # Set of parameters to go through with gridsearch
    parameters = {
        # 'vect__ngram_range': ((1, 1), (1, 2)),
        # 'vect__max_df': (0.5, 1.0),
        # 'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [20, 30],
    }

    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2)
    return model


def evaluate_model(model, X_test, y_test, category_names):
    """
    Prints out the model's performance on the test data to the consoleself.
    Includes precision, recall, f-1 score, and support size.

    Args:
        model - Trained sklearn model with predict method
        X_test - pd.DataFrame with test features
        y_test - pd.DataFrame with test target values
        category_names - List of strings with category names
    """
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)
    print(classification_report(y_test, y_pred, target_names=category_names))


def save_model(model, model_filepath):
    """
    Saves a given model to the provided filepath as a pickel file.

    Args:
        model - model to be saved
        model_filepath - path and filename (including extension) to save the model
    """
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)


def main():
    """
    Main function to orchestrate the training script.
    """

    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

        print('Building model...')
        model = build_model()

        print('Training model...')
        model.fit(X_train, Y_train)

        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '
              'as the first argument and the filepath of the pickle file to '
              'save the model to as the second argument. \n\nExample: python '
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
