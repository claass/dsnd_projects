import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_path, categories_path):
    messages = pd.read_csv(messages_path)
    categories = pd.read_csv(categories_path)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    # Split the joint categories column into individual ones
    categories = df.categories.str.split(";", expand=True)

    # Rename column names
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames

    # Binarize column values
    for column in categories:
        categories[column] = categories[column].str[-1]
        categories[column] = categories[column].astype(int)

    # Replace the old categories column
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    # Return dataframe
    return df


def save_data(df, database_filename):
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterReliefTweets', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_path, categories_path, database_path = sys.argv[1:]

        print('Loading...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_path, categories_path))
        df = load_data(messages_path, categories_path)

        print('Cleaning...')
        df = clean_data(df)

        print('Saving...\n    DATABASE: {}'.format(database_path))
        save_data(df, database_path)

        print('Cleaned data saved to database!')

    else:
        print('Please provide the filepaths of the messages and categories '
              'datasets as the first and second argument respectively, as '
              'well as the filepath of the database to save the cleaned data '
              'to as the third argument. \n\nExample: python process_data.py '
              'disaster_messages.csv disaster_categories.csv '
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
