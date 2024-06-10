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
    """Automatically detect and convert (in place!) each
    dataframe column of datatype 'object' to a datetime just
    when ALL of its non-NaN values can be successfully parsed
    by pd.to_datetime().  Also returns a ref. to df for
    convenient use in an expression.
    """
    for c in df.columns[df.dtypes=='object']:
        try:
            df[c]=pd.to_datetime(df[c])
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
#orders_df = convert_datatype(orders_df)

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
    # Ensure the date column is in datetime format
    strfttime = get_time_format(data[date_column].iloc[0]).choices[0]["message"]["content"]
    data[date_column] = pd.to_datetime(data[date_column], format=strfttime)
    data[customer_id_column] = data[customer_id_column].astype(str)
    # Get the maximum date in your data -- this will be considered as the current date for the analysis
    current_date = data[date_column].max()

    # Calculate Recency, Frequency, and Monetary value for each customer
    rfm = data.groupby(customer_id_column).agg({
        date_column: lambda x: (x.max() - current_date).days,  # Recency: Days since last purchase
        customer_id_column: 'count',  # Frequency: Total number of purchases
        monetary_value_column: 'sum'  # Monetary: Total monetary value
    })

    # Rename the columns
    rfm.rename(columns={
        date_column: 'Recency',
        customer_id_column: 'Frequency',
        monetary_value_column: 'MonetaryValue'
    }, inplace=True)
    return rfm

def custom_quantiles(rfm, r_bins, f_bins):
    # Here's the tricky part. We need to handle the situation where there are not enough unique quantiles.
    r_quantiles_list = [(x + 1) / (r_bins) for x in range(0, r_bins)]
    f_quantiles_list = [(x + 1) / (f_bins) for x in range(0, f_bins)]
    r_quantiles = rfm.quantile(q=r_quantiles_list)
    f_quantiles = rfm.quantile(q=f_quantiles_list)

    # We apply a custom function to assign the score based on the defined quantiles.
    # The 'searchsorted' method is used to label each quantile.
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

    # We apply the scoring to R, F, and M.
    rfm['R'] = rfm['Recency'].apply(lambda input: rfm_scoring(input, r_bins, r_quantiles_list, 'Recency', r_quantiles))
    rfm['F'] = rfm['Frequency'].apply(lambda input: rfm_scoring(input, f_bins, f_quantiles_list, 'Frequency', f_quantiles))

    # Calculate a composite RFM score
    return rfm

rfm_response = openai.ChatCompletion.create(
        model="gpt-4-1106-preview",
        messages= [
            {"role": "system",
             "content": "You are a data analyst that is an expert in providing RFM Analysis. You will be given the first few rows of a dataframe. Run functions that you have been provided with. Only use the functions you have been provided with"},
            {"role": "system",
             "content": f"This is the first few rows of your dataframe: \n{orders_df.head()}"},
        ],
        functions = [
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
        st.write(function_response)
        st.scatter_chart(data=function_response, x="Recency", y="Frequency", color="MonetaryValue")
        r_iqr = abs(function_response["Recency"].quantile(0.75) - function_response["Recency"].quantile(0.25))
        r_bin_width = math.ceil(r_iqr / 2 + 0.00001)
        f_iqr = function_response["Frequency"].quantile(0.75) - function_response["Frequency"].quantile(0.25)
        f_bin_width = math.ceil(f_iqr / 2 + 0.00001)
        r_slider = st.slider('What is the bin width you would like for recency?', 1, 300, r_bin_width)
        f_slider = st.slider('What is the bin width you would like for frequency?', 1, 20, f_bin_width)
        r_max_suggest = function_response["Recency"].quantile(0.25) - 1.5 * r_iqr
        f_max_suggest = function_response["Frequency"].quantile(0.75) + 1.5 * f_iqr
        r_max_slider = st.slider('What is the min you would like for recency?', -500, 0, math.ceil(r_max_suggest + 0.00001))
        f_max_slider = st.slider('What is the max you would like for frequency?', 1, 20, math.ceil(f_max_suggest + 0.00001))
        #fig = px.density_heatmap(function_response, x="Recency", y="Frequency", z="MonetaryValue", histfunc="avg", nbinsx=r_slider, nbinsy=f_slider, color_continuous_scale="Tealgrn", text_auto=True)
        fig = go.Figure(go.Histogram2d(x=function_response["Recency"], y=function_response["Frequency"], histfunc="count",
                                       xbins = dict(start=r_max_slider, end=0, size=r_slider),
                                       ybins = dict(start=0, end=f_max_slider, size=f_slider),
                                       texttemplate="%{z}"
                                       ))
        fig.update_layout(
            xaxis_title='Recency',
            yaxis_title='Frequency',
            bargap=0.1,  # Controls the gap between bars
        )
        st.title("Number of customers")
        st.plotly_chart(fig, use_container_width=True)
        st.title("AOV")
        fig2 = go.Figure(go.Histogram2d(x=function_response["Recency"], y=function_response["Frequency"],
                                        z=function_response["MonetaryValue"], histfunc="avg",
                                        xbins=dict(start=r_max_slider, end=0, size=r_slider),
                                        ybins=dict(start=0, end=f_max_slider, size=f_slider),
                                        texttemplate="%{z}"
                                        ))
        fig2.update_layout(
            xaxis_title='Recency',
            yaxis_title='Frequency',
            bargap=0.1,  # Controls the gap between bars
        )
        st.plotly_chart(fig2, use_container_width=True)
        import pandas as pd
        import numpy as np

        # Assuming you have a DataFrame `df` with 'Recency', 'Frequency', and 'Monetary' columns
        # df = pd.DataFrame({
        #     'Recency': [...],   # Your recency data
        #     'Frequency': [...], # Your frequency data
        #     'Monetary': [...]   # Your monetary data
        # })


        # Calculate the bin edges based on the minimum and maximum values and the bin width
        recency_bins = np.arange(start=r_max_slider, stop=r_slider,
                                 step=r_slider)
        frequency_bins = np.arange(start=0, stop=f_max_slider + f_slider,
                                   step=f_slider)

        # Use numpy's histogram2d function to calculate the bin counts
        hist, xedges, yedges = np.histogram2d(x=function_response['Frequency'], y=function_response['Recency'], bins=[frequency_bins, recency_bins])

        # Now `hist` contains the counts for each bin
        # If you need to work with the counts in a DataFrame
        hist_df = pd.DataFrame(hist, index=xedges[:-1], columns=yedges[:-1])
        hist_df_original = hist_df
        hist_df = hist_df.loc[:, (hist_df != 0).any(axis=0)]
        # Step 2: Drop rows with all zeros
        hist_df = hist_df.loc[(hist_df != 0).any(axis=1)]

        #st.write(hist_df_original)
        st.write(hist_df)
        get_gpt_segments = openai.ChatCompletion.create(
            model="gpt-4-1106-preview",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system",
                 "content": f"""If I had a dataframe below like this: \n {hist_df} \n where the headers represents the number of days since the customer last purchased, displayed in negative numbers. The less negative they are, the more recently they purchased. (Recency), and the index represents the number of times the customers purchased (frequency), come up with DISTINCT AND NON-OVERLAPPING segmentation conditions for me as a marketer, where the datum in this dataframe is more than 0 \
Please name these segments creatively if this company is {company_desc}. Return the segments in a json format like the following WITHOUT ANY EXPLANATION: \
{{"segments": [{{"name": "Segment Name", "desc": "Description of this segment", "r_condition": "Recency condition to be added to df.query(). Use Recency as the header.", "f_condition": "Frequency condition to be added to df.query(). Use Frequency as the header. "}}]}}\
    """},
            ],
            temperature=0.7
        )
        st.title("GPT Suggested Segments")
        #st.write(get_gpt_segments["choices"][0]["message"])
        for x in json.loads(get_gpt_segments["choices"][0]["message"]["content"])["segments"]:
            try:
                st.subheader(f"""{x["name"]} \n {x["r_condition"]}, {x["f_condition"]} """, divider='rainbow')
                st.write(f"""Description of segment: {x["desc"]}""")
                st.write("Number of contacts: ", len(function_response.query(f"""({x["r_condition"]}) & ({x["f_condition"]})""")))
                st.write(function_response.query(f"""({x["r_condition"]}) & ({x["f_condition"]})"""))
            except:
                None
