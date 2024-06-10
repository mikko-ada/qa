from mlxtend.frequent_patterns import fpgrowth, association_rules
from mlxtend.preprocessing import TransactionEncoder
import openai
import streamlit as st
from io import StringIO
import re
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
import json
import math

openai.api_key = "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"

st.title("Market Basket Analysis")

def rename_dataset_columns(dataframe):
    dataframe.columns = dataframe.columns.str.replace('[#,@,&,$,(,)]', '')
    dataframe.columns = [re.sub(r'%|_%', '_percentage', x) for x in dataframe.columns]
    dataframe.columns = dataframe.columns.str.replace(' ', '_')
    dataframe.columns = [x.lstrip('_') for x in dataframe.columns]
    dataframe.columns = [x.strip() for x in dataframe.columns]
    return dataframe


def convert_datatype(df):
    """Automatically detect and convert (in place!) each
    dataframe column of datatype 'object' to a datetime just
    when ALL of its non-NaN values can be successfully parsed
    by pd.to_datetime().  Also returns a ref. to df for
    convenient use in an expression.
    """
    for c in df.columns[df.dtypes == 'object']:
        try:
            df[c] = pd.to_datetime(df[c])
        except:
            print("None")

    df = df.convert_dtypes()
    return df


uploaded_files = st.file_uploader("Choose a CSV file", accept_multiple_files=True)


@st.cache_data
def load_data(files):
    for uploaded_file in files:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        orders_df = pd.read_csv(stringio)
        return orders_df


orders_df = load_data(uploaded_files)
orders_df = rename_dataset_columns(orders_df)


# orders_df = convert_datatype(orders_df)

def get_time_format(time):
    return openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages=[
            {"role": "system",
             "content": f"""If I had a datetime sting like this: {time}, what is the strftime format of this string? Return the strfttime format only. Do not return anything else."""},
        ],
        temperature=0
    )

MBA_headers = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages= [
            {"role": "system",
             "content": "You are a data analyst that is an expert in providing RFM Analysis. You will be given the first few rows of a dataframe. Run functions that you have been provided with. Only use the functions you have been provided with"},
            {"role": "system",
             "content": f"This is the first few rows of your dataframe: \n{orders_df.head()}"},
        ],
        functions = [
                {
                    "name": "get_MBA",
                    "description": "Create a market basket analysis using the Apiori algorithm. This functions accepts the order ID and the product name as the arguments to run this analysis.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "line_item_name_column": {
                                "type": "string",
                                "description": f"The name of the column header in the provided dataframe that contains the name of the product or line item. It must be one of {orders_df.columns.tolist()}",
                            },
                            "order_id_column": {
                                "type": "string",
                                "description": f"The name of the column header in the provided dataframe that contains value that uniquely identifies an order. It is typically numeric or alphanumeric. It must be one of {orders_df.columns.tolist()}",
                            },

                        },
                        "required": ["order_id_column", "line_item_name_column"],
                    },
                }
            ],
        function_call={"name": "get_MBA"},
        temperature=0
    )

st.write(MBA_headers)

suggested_lineitem = [i for i in range(len(orders_df.columns)) if orders_df.columns.tolist()[i] == json.loads(MBA_headers.choices[0]["message"]["function_call"]["arguments"]).get("line_item_name_column")]
suggested_orderid = [i for i in range(len(orders_df.columns)) if orders_df.columns.tolist()[i] == json.loads(MBA_headers.choices[0]["message"]["function_call"]["arguments"]).get("order_id_column")]

col_for_lineitem = st.selectbox(
    'What is the column used to identify name of a line item?',
    orders_df.columns,
    index=suggested_lineitem[0]
)

col_for_orderid = st.selectbox(
    'What is the column used for order ID?',
    orders_df.columns,
    index=suggested_orderid[0]
)

def get_MBA(order_id_column, line_item_name_column):
    # Step 1: Preprocess data to create a list of items per order
    order_items = orders_df.groupby(order_id_column)[line_item_name_column].apply(list).reset_index()
    # Step 2: Transaction Encoding
    transaction_encoder = TransactionEncoder()
    transaction_array = transaction_encoder.fit(order_items[line_item_name_column]).transform(order_items[line_item_name_column])
    transaction_df = pd.DataFrame(transaction_array, columns=transaction_encoder.columns_)
    # Step 3: Apriori Algorithm to find frequent itemsets
    frequent_itemsets = fpgrowth(transaction_df, min_support=0.01, use_colnames=True)
    # Step 4: Association Rules
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.01)
    # You can sort the rules by confidence, lift, etc.
    rules_sorted = rules.sort_values(by='lift', ascending=False)
    # Display the top 10 rules
    return (rules_sorted)

st.write(get_MBA(order_id_column=col_for_orderid, line_item_name_column=col_for_lineitem))
