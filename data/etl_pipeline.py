#!/usr/bin/env python
# coding: utf-8

# import libraries
import pandas as pd
from sqlalchemy import create_engine



def process_data(message_file_name, category_file_name, db_file_name):
    """
    This is the function that generates database file from message file and category file.

    :param message_file_name: message file name
    :param category_file_name: category file name
    :param db_file_name: data base file name
    :return: an exception will be threw if error happens
    """
    # step 1: merge two csv files into one data frame
    messages = pd.read_csv(message_file_name)
    categories = pd.read_csv(category_file_name)
    df = messages.merge(categories, on='id')

    # step 2: reorganize the categories column
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = [category_name.split('-')[0] for category_name in row]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].astype(str).str[-1]
        categories[column] = categories[column].astype(int)

    new_df = df.drop(['categories'], axis=1)
    df = pd.concat([new_df, categories],  axis=1)

    # step 3: remove duplicate records
    duplicated_record_num = df.duplicated().sum()
    cleaned_df = df.drop_duplicates()
    assert cleaned_df.duplicated().sum() == 0

    # step 4: generate the database
    table_name = "disaster_response_table"
    engine = create_engine('sqlite:///' + db_file_name)
    cleaned_df.to_sql(table_name, engine, index=False)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Generate processed data base')
    parser.add_argument('--message', metavar='path', required=False,
                        help='the path to message csv file')
    parser.add_argument('--category', metavar='path', required=False,
                        help='path to category csv file')
    parser.add_argument('--database', metavar='path', required=False,
                        help='path to .db database')
    args = parser.parse_args()

    message_file_name = args.message
    category_file_name = args.category
    db_file_name = args.database

    if message_file_name is None:
        message_file_name = "messages.csv"
    if category_file_name is None:
        category_file_name = "categories.csv"
    if db_file_name is None:
        db_file_name = "DisasterResponse.db"

    process_data(message_file_name, category_file_name, db_file_name)








	









