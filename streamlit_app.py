# First
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

def rename_dataset_columns(dataframe):
    dataframe.columns = dataframe.columns.str.replace('[#,@,&,$,(,)]', '')
    dataframe.columns = [re.sub(r'%|_%', '_percentage', x) for x in dataframe.columns]
    dataframe.columns = dataframe.columns.str.replace(' ', '_')
    dataframe.columns = [x.lstrip('_') for x in dataframe.columns]
    dataframe.columns = [x.strip() for x in dataframe.columns]
    return dataframe

def convert_datatype(df):
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

def rfm_analysis(date_column, customer_id_column, monetary_value_column):
    data = orders_df
    data = data.dropna(subset=[date_column, customer_id_column, monetary_value_column])
    strfttime = get_time_format(data[date_column].iloc[0]).choices[0]["message"]["content"]
    data[date_column] = pd.to_datetime(data[date_column], format=strfttime)
    data[customer_id_column] = data[customer_id_column].astype(str)
    current_date = data[date_column].max()
    rfm = data.groupby(customer_id_column).agg({
        date_column: lambda x: (x.max() - current_date).days,
        customer_id_column: 'count',
        monetary_value_column: 'sum'
    })
    rfm.rename(columns={
        date_column: 'Recency',
        customer_id_column: 'Frequency',
        monetary_value_column: 'MonetaryValue'
    }, inplace=True)
    return rfm

def custom_quantiles(rfm, r_bins, f_bins):
    r_quantiles_list = [(x + 1) / (r_bins) for x in range(0, r_bins)]
    f_quantiles_list = [(x + 1) / (f_bins) for x in range(0, f_bins)]
    r_quantiles = rfm.quantile(q=r_quantiles_list)
    f_quantiles = rfm.quantile(q=f_quantiles_list)

    def rfm_scoring(input, bins, quantile_list, parameter, quantiles):
        value = ""
        if parameter == "Recency":
            for q in reversed(range(len(quantile_list))):
                if input <= quantiles[parameter][quantile_list[q]]:
                    value = q + 1
                else:
                    break
            return value
        elif parameter == "Frequency":
            for q in reversed(range(0, len(quantile_list))):
                if input <= quantiles[parameter][quantile_list[q]]:
                    value = q + 1
                else:
                    break
            return value

    rfm['R'] = rfm['Recency'].apply(lambda input: rfm_scoring(input, r_bins, r_quantiles_list, 'Recency', r_quantiles))
    rfm['F'] = rfm['Frequency'].apply(lambda input: rfm_scoring(input, f_bins, f_quantiles_list, 'Frequency', f_quantiles))

    return rfm

rfm_response = openai.ChatCompletion.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "system",
         "content": "You are a data analyst that is an expert in providing RFM Analysis. You will be given the first few rows of a dataframe. Run functions that you have been provided with. Only use the functions you have been provided with"},
        {"role": "system",
         "content": f"This is the first few rows of your dataframe: \n{orders_df.head()}"},
    ],
    functions=[
        {
            "name": "rfm_analysis",
            "description": "Create an RFM analysis in an orders dataset",
            "parameters": {
                "type": "object",
                "properties": {
                    "date_column": {
                        "type": "string",
                        "description": f"The name of the column header in the provided dataframe that contains the datestrings in which an order is being placed. It must be one of {orders_df.columns.tolist()}",
                    },
                    "customer_id_column": {
                        "type": "string",
                        "description": f"The name of the column header in the provided dataframe that contains value that uniquely identifies a customer. The column typically contains an email or a form of id. It must be one of {orders_df.columns.tolist()}",
                    },
                    "monetary_value_column": {
                        "type": "number",
                        "description": f"The name of the column header in the provided dataframe that contains the amount spent per order. The column name typically contains the string Total or Amount. It must be one of {orders_df.columns.tolist()}",
                    }
                },
                "required": ["date_column", "customer_id_column", "monetary_value_column"],
            },
        }
    ],
    function_call={"name": "rfm_analysis"},
    temperature=0
)

suggested_r = [i for i in range(len(orders_df.columns)) if orders_df.columns.tolist()[i] == json.loads(rfm_response.choices[0]["message"]["function_call"]["arguments"]).get("date_column")]
suggested_f = [i for i in range(len(orders_df.columns)) if orders_df.columns.tolist()[i] == json.loads(rfm_response.choices[0]["message"]["function_call"]["arguments"]).get("customer_id_column")]
suggested_m = [i for i in range(len(orders_df.columns)) if orders_df.columns.tolist()[i] == json.loads(rfm_response.choices[0]["message"]["function_call"]["arguments"]).get("monetary_value_column")]

col_for_r = st.selectbox(
    'What is the column used for recency - date of purchase?',
    orders_df.columns,
    index=dict(enumerate(suggested_r)).get(0, 0)
)

col_for_f = st.selectbox(
    'What is the column used to identify a customer ID?',
    orders_df.columns,
    index=dict(enumerate(suggested_f)).get(0, 0)
)

col_for_m = st.selectbox(
    'What is the column used for order value?',
    orders_df.columns,
    index=dict(enumerate(suggested_m)).get(0, 0)
)

st.title("Key metrics")
if rfm_response.choices[0]["message"].get("function_call"):
    function_args = json.loads(rfm_response.choices[0]["message"]["function_call"]["arguments"])
    function_response = rfm_analysis(
        date_column=col_for_r,
        customer_id_column=col_for_f,
        monetary_value_column=col_for_m
    )
    if function_response is not None:
        st.metric(label="Average spending per customer", value=function_response["MonetaryValue"].mean())
        st.metric(label="Average number of purchase per customer", value=function_response["Frequency"].mean())
        st.metric(label="Average order value", value=orders_df[col_for_m].mean())

st.title("Buying frequency")
if rfm_response.choices[0]["message"].get("function_call"):
    function_args = json.loads(rfm_response.choices[0]["message"]["function_call"]["arguments"])
    function_response = rfm_analysis(
        date_column=col_for_r,
        customer_id_column=col_for_f,
        monetary_value_column=col_for_m
    )
    if function_response is not None:
        st.write(function_response[["Frequency", "Recency"]])
        fig = px.histogram(function_response, x="Frequency")
        st.write(fig)

st.title("RFM Analysis")
company_desc = st.text_input('Description of company', 'an ecommerce company selling goods')

if rfm_response.choices[0]["message"].get("function_call"):
    function_args = json.loads(rfm_response.choices[0]["message"]["function_call"]["arguments"])
    function_response = rfm_analysis(
        date_column=col_for_r,
        customer_id_column=col_for_f,
        monetary_value_column=col_for_m
    )
    if function_response is not None:
        r_iqr = abs(function_response["Recency"].quantile(0.75) - function_response["Recency"].quantile(0.25))
        r_bin_width = math.ceil(r_iqr / 2 + 0.00001)
        r_bins = (function_response["Recency"].max() - function_response["Recency"].min()) // r_bin_width
        f_iqr = abs(function_response["Frequency"].quantile(0.75) - function_response["Frequency"].quantile(0.25))
        f_bin_width = math.ceil(f_iqr / 2 + 0.00001)
        f_bins = (function_response["Frequency"].max() - function_response["Frequency"].min()) // f_bin_width

        rfm_ = custom_quantiles(function_response, r_bins, f_bins)

        st.write(rfm_)

        fig = px.scatter(rfm_, x="Recency", y="Frequency", color="R")
        fig.update_layout(title="Recency vs Frequency by Recency score")
        st.plotly_chart(fig)

        fig = px.scatter(rfm_, x="Recency", y="Frequency", color="F")
        fig.update_layout(title="Recency vs Frequency by Frequency score")
        st.plotly_chart(fig)

