import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_path, categories_path):
    """
    Loads messages and categories from respective csv files and returns
    a combined dataframe.

    Args:
        messages_path (str): path to messages input csv
        categories_path (str): path to categories input csv

    Returns:
        Pandas DataFrame with the combined data sources
    """
    messages = pd.read_csv(messages_path)
    categories = pd.read_csv(categories_path)
    df = pd.merge(messages, categories, on='id')
    return df


def clean_data(df):
    """
    Cleans the given dataframe (see inline comments for individual steps)
    and returns the cleaned DataFrame

    Args:
        df (Pandas DataFrame): Dataframe in need of cleaning

    Returns:
        Cleaned DataFrame
    """

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

    # Fix improperly labeled observations in the 'related' column
    categories.related.replace(2, 0, inplace=True)

    # Replace the old categories column
    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)

    # Remove duplicates
    df = df.drop_duplicates()

    # Return dataframe
    return df


def save_data(df, database_filename):
    """
    Saves the dataframe as a sqlite database to the given path.
    Table name is "Tweets". Schema as defined by the input dataframe.

    Args:
        df (Pandas DataFrame): DataFrame to be saved
        database_filename (str): Path and name with extension of the database
    """
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Tweets', engine, index=False)


def main():
    """
    Main function to orchestrate the cleaning process.
    """
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
