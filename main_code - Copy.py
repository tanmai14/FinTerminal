# import time
import dash
import concurrent.futures
from dash import Dash
import dash_mantine_components as dmc
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go
import plotly.express as px
import pandas as pd
import numpy as np
import requests
import json

# KEY = "" Your Darqube API KEY here

# KEY_Alpha = Your AlphaVantage Key Here

f = pd.read_csv('./ticker - Copy.csv')
data = np.array(f['ticker'])

comp = np.array(f['name'])


data_list = []
label_data = ['Revenue', 'Operating Expenses', 'Gross Profit', 'Net Income', 'EBIT', 'EPS', 'Free Cash Flow',
              'Cash', 'Debt', 'Cash - Operating Activities',
              'Cash - Investing Activities', 'Cash - Financing Activities']

value_data = ['I_totalRevenue', 'I_totalOperatingExpenses', 'I_grossProfit', 'I_netIncome', 'I_ebit', 'EPS', 'C_freeCashFlow',
              'B_cash', 'B_netDebt', 'C_totalCashFromOperatingActivities', 'C_totalCashflowsFromInvestingActivities', 'C_totalCashFromFinancingActivities']

for i, j in zip(data, comp):
    data_list.append({'label': i+" "+"({})".format(j), 'value': i})



app = Dash(__name__, external_stylesheets=[
           'https://fonts.googleapis.com/css2?family=Inconsolata'])

app.layout = html.Div(style={'margin-left': '20px'},
                      children=[

    html.H1("FinTerminal", className="Heading", id='head'),

    html.Br(),

    html.Div(className="Dropdown fixed-header",
             children=[dcc.Dropdown(options=data_list, id='ticker-search-dropdown', value='AAPL', clearable=True)], style={'display': 'block'}),

    html.Br(),

    html.Div(id='1', style={'width': '100%'}, children=[

        html.Iframe(id='Frame', srcDoc=None,
                    style={'width': '100%',
                           'height': '520px', 'border': 'none'}
                    )]),

    dmc.Table(id='card', horizontalSpacing=1, children=[

    ]),
    html.Br(),
    html.Br(),


    # Code for charts of fundamentals
    
    html.Div([
        html.H1('Company Financials', style={'text-align': 'center'}),

        dcc.RadioItems(
            id='timeline',
            options=[
                {'label': 'Quarterly', 'value': 'quarterly'},
                {'label': 'Annual', 'value': 'yearly'}
            ],
            value='quarterly',
            style={'justify-content': 'right', 'margin': '20px', 'text-align': 'right', 'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px', 'width': 'fit-content'}),


        dcc.Checklist(
            id='area-checklist',
            options=[
                {'label': str(x)+" \t\t", 'value': y} for x, y in zip(label_data, value_data)
            ],
            value=[],  # Default checked items
            style={'margin': '20px'},
        ),

        html.Div(
            id='clear-check',
            style={'justify-content': 'right', 'margin': '20px',
                   'text-align': 'right', 'padding': '10px', 'width': 'fit-content'},
            children=[
                html.Button("Select All", id='button1', n_clicks=0),
                html.Button("Clear All", id='button2', n_clicks=0)
            ]),

        html.Br(),

        dcc.Store(id='selected-areas'),

        html.P('Additional Parameters from SEC Filings',
               style={'margin': '20px'}),
        dmc.MultiSelect(id='SEC',
                        data=[],
                        value=[],
                        searchable=True,
                        clearable=True,
                        nothingFound="No options found",
                        placeholder="Search for Advanced Features straight from SEC Filings!",
                        style={"width": 400, 'margin': '20px'},
                        ),

        html.Hr(),
        html.Div(id='area-plots',
                 style={'display': 'flex', 'flexWrap': 'wrap'})
    ],
        style={'padding': '10px', 'border': '1px solid #ccc', 'border-radius': '5px'})
]
)

# Code for TradingView Graph setup

@app.callback(
    [Output('Frame', 'srcDoc'),
     Output('area-checklist', 'value')],
    [Input("ticker-search-dropdown", 'value'),
     Input('button1', 'n_clicks'),
     Input('button2', 'n_clicks')]
)
def retrive_name(ticker, b1, b2):
    # print(ticker)
    if ticker != None:
        # print(ticker)
        global tick
        tick = ticker
        tradingview_code = f'''
        <div class="tradingview-widget-container">
            <div id="tradingview_7dc43"></div>
            <div class="tradingview-widget-copyright">
                <a href="https://in.tradingview.com/" rel="noopener nofollow" target="_blank">
                    <span class="blue-text">Track all markets on TradingView</span>
                </a>
            </div>
            <script type="text/javascript" src="https://s3.tradingview.com/tv.js"></script>
            <script type="text/javascript">
                new TradingView.widget(
                {{
                    "autosize": false,
                    "symbol": "{ticker}",
                    "interval": "D",
                    "timezone": "exchange",
                    "theme": "dark",
                    "style": "2",
                    "locale": "in",
                    "enable_publishing": false,
                    "backgroundColor": "rgba(0, 0, 0, 1)",
                    "hide_top_toolbar": true,
                    "save_image": false,
                    "calendar": false,
                    "container_id": "tradingview_7dc43"
                }});
            </script>
        </div>
    '''
    print("Hi1")
    ctx = dash.callback_context
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    # print(button_id)
    s = []
    if button_id == 'button1':
        s = value_data
    if button_id == 'button2':
        s = []
    return tradingview_code, s


# Code for SEC API

@app.callback(
    Output('SEC', 'data'),
    Input("ticker-search-dropdown", 'value')
)
def sec_call(ticker):
    global CIK
    global sec_d
    CIK = np.array(f[f['ticker'] == ticker]['CIK'])[0]
    res = requests.get("https://data.sec.gov/api/xbrl/companyfacts/{}.json".format(
        # 'CIK'+str(str(CIK).zfill(10))), headers={'User-Agent': 'Name email'}) Include your name and email to access SEC data through EDGAR
    print(res)
    sec_d = res.json()['facts']['us-gaap']
    sec_lab = list(map(camel_case_to_title_case, list(sec_d.keys())))
    sec_val = sec_d.keys()
    sec_data = [
        {'value': i, 'label': j} for i, j in zip(sec_val, sec_lab)
    ]
    return sec_data

# Code for data retrival through API

@app.callback(
    Output('card', 'children'),
    Input("ticker-search-dropdown", 'value')
)
def api_call(ticker):
    # print('Hellooo')
    global df_I, df_B, df_C, df_EPS, df_IN, df_DIV, df_OUT
    # print(CIK)
    urls = [
        "https://api.darqube.com/data-api/fundamentals/stocks/income_statement/{}?token={}".format(
            ticker, KEY),
        "https://api.darqube.com/data-api/fundamentals/stocks/balance_sheet/{}?token={}".format(
            ticker, KEY),
        "https://api.darqube.com/data-api/fundamentals/stocks/cash_flow/{}?token={}".format(
            ticker, KEY),
        "https://api.darqube.com/data-api/fundamentals/stocks/institutional_holders/{}?token={}".format(
            ticker, KEY),
        "https://api.darqube.com/data-api/fundamentals/stocks/eps_historical/{}?token={}".format(
            ticker, KEY),
        "https://api.darqube.com/data-api/fundamentals/stocks/outstanding_shares/{}?token={}".format(
            ticker, KEY),
        "https://api.darqube.com/data-api/fundamentals/stocks/dividends/{}?token={}".format(
            ticker, KEY),
        "https://www.alphavantage.co/query?function=OVERVIEW&symbol={}&apikey=WA7IMVZJ6I6RNCDR".format(
            ticker),
    ]
    # print("hesgrdhfgjghc")
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(urls)) as executor:
        results = list(executor.map(fetch_data, urls))

    df_I, df_B, df_C, df_IN, df_EPS, df_OUT, df_DIV, a = results

    for i in ['CIK', 'AssetType', 'Exchange', 'Description', 'Currency', 'Country', 'Address', 'FiscalYearEnd', 'LatestQuarter']:
        del a[i]
    rows = []
    ele = []
    count = 0
    row = None
    for record in a.keys():
        if count < 3:
            k = record
            val = a[k]
            ele.extend([html.Td(style={
                       'color': 'lightgrey'}, children=camel_case_to_title_case(k)), html.Td(val)])
            count += 1
        else:
            count = 0
            row = html.Tr(ele)
            ele = []
            rows.append(row)

    table_content = html.Tbody(rows)

    return table_content


@app.callback(
    Output('selected-areas', 'data'),
    Input('area-checklist', 'value')
)
def store_selected_areas(selected_areas):
    return selected_areas


# Code for Graphs and Visualizations

@app.callback(
    Output('area-plots', 'children'),
    [Input('selected-areas', 'data'),
     Input('timeline', 'value'), Input("ticker-search-dropdown", 'value'), Input('SEC', 'value')]
)
def update_plots(selected_areas, timeline, ticker, sec_values):
    # time.sleep(0.3)
    plots = []
    row = []
    for idx, area in enumerate(selected_areas):

        # For Income Statement Values
        if 'I_' in area:
            t = camel_case_to_title_case(area[2:])
            temp = pd.read_json(json.dumps(df_I[timeline])).transpose()
            # temp.dropna(inplace=True)
            # print(temp)
            fig = px.area(temp, x=temp.index,
                          y=area[2:], title=t, width=450, height=400)
            text_color = '#FFFFFF'
            fig.update_layout(paper_bgcolor='#1C1C1C', plot_bgcolor='black', margin=dict(
                r=10, l=0), title_text=t+' of '+ticker, title_x=0.5, title_font=dict(color=text_color), xaxis_title='Date',                  # Set title text color
                # Set x-axis title and labels color
                xaxis=dict(title_font=dict(color=text_color)),
                # Set y-axis title and labels color
                yaxis=dict(title_font=dict(color=text_color)),
                # Set legend title and labels color
                legend=dict(title_font=dict(color=text_color)),
                font=dict(color=text_color))
            fig.update_xaxes(mirror=True, ticks='outside', showline=True,
                             linecolor='black', showgrid=True, gridcolor='#292829', gridwidth=1,)
            fig.update_yaxes(mirror=True, ticks='outside', showline=True,
                             showgrid=True, gridcolor='#292829', gridwidth=1, linecolor='black',)
            fig.update_traces(line=dict(color='#ffa028'))
            graph = dcc.Graph(figure=fig, style={
                              'flex': '50%', 'margin': '10px'})
            row.append(graph)

        # For Balance Sheet
        if 'B_' in area:
            t = camel_case_to_title_case(area[2:])
            temp = pd.read_json(json.dumps(df_B[timeline])).transpose()
            fig = px.area(temp, x=temp.index,
                          y=area[2:], title=t, width=450, height=400)
            text_color = '#FFFFFF'
            fig.update_layout(paper_bgcolor='#1C1C1C', plot_bgcolor='black', margin=dict(
                r=10, l=0), title_text=t+' of '+ticker, title_x=0.5, title_font=dict(color=text_color), xaxis_title='Date',                  # Set title text color
                # Set x-axis title and labels color
                xaxis=dict(title_font=dict(color=text_color)),
                # Set y-axis title and labels color
                yaxis=dict(title_font=dict(color=text_color)),
                # Set legend title and labels color
                legend=dict(title_font=dict(color=text_color)),
                font=dict(color=text_color))
            fig.update_xaxes(mirror=True, ticks='outside', showline=True,
                             linecolor='black', showgrid=True, gridcolor='#292829', gridwidth=1,)
            fig.update_yaxes(mirror=True, ticks='outside', showline=True,
                             showgrid=True, gridcolor='#292829', gridwidth=1, linecolor='black',)
            fig.update_traces(line=dict(color='#ffa028'))
            graph = dcc.Graph(figure=fig, style={
                              'flex': '50%', 'margin': '10px'})
            row.append(graph)

        # For Cash Statement

        if 'C_' in area:
            t = camel_case_to_title_case(area[2:])
            temp = pd.read_json(json.dumps(df_C[timeline])).transpose()
            fig = px.area(temp, x=temp.index,
                          y=area[2:], title=t, width=450, height=400)
            text_color = '#FFFFFF'
            fig.update_layout(paper_bgcolor='#1C1C1C', plot_bgcolor='black', margin=dict(
                r=10, l=0), title_text=t+' of '+ticker, title_x=0.5, title_font=dict(color=text_color), xaxis_title='Date',                  # Set title text color
                # Set x-axis title and labels color
                xaxis=dict(title_font=dict(color=text_color)),
                # Set y-axis title and labels color
                yaxis=dict(title_font=dict(color=text_color)),
                # Set legend title and labels color
                legend=dict(title_font=dict(color=text_color)),
                font=dict(color=text_color))
            fig.update_xaxes(mirror=True, ticks='outside', showline=True,
                             linecolor='black', showgrid=True, gridcolor='#292829', gridwidth=1,)
            fig.update_yaxes(mirror=True, ticks='outside', showline=True,
                             showgrid=True, gridcolor='#292829', gridwidth=1, linecolor='black',)
            fig.update_traces(line=dict(color='#ffa028'))
            graph = dcc.Graph(figure=fig, style={
                              'flex': '50%', 'margin': '10px'})
            row.append(graph)

        if 'EPS' in area:
            text_color = '#FFFFFF'
            data = json.dumps(df_EPS)
            eps = pd.read_json(data).transpose(
            ).loc[:, ('epsActual', 'epsEstimate', 'surprisePercent')]
            eps = eps.sort_index().iloc[-15:-1, :]
            # eps=eps.replace({
            #     None:0
            # })
            eps.dropna(inplace=True)
            # eps['Marker_Size'] = eps.apply(calculate_marker_size, axis=1)
            eps['Marker_Color_Actual'] = eps.apply(
                calculate_marker_color_actual, axis=1)
            eps['Marker_Color_Estimate'] = 'white'
# Update the graph directly without using a callback
            fig = px.scatter(eps, x=eps.index, y=[
                             'epsActual', 'epsEstimate'], title='EPS Actual vs Estimates', opacity=1.0, width=500, height=400)

            fig.update_layout(paper_bgcolor='#1C1C1C', plot_bgcolor='black', xaxis=dict(title_font=dict(color=text_color)),      # Set x-axis title and labels color
                              # Set y-axis title and labels color
                              yaxis=dict(title_font=dict(color=text_color)),
                              # Set legend title and labels color
                              legend=dict(title_font=dict(color=text_color)),
                              font=dict(color=text_color), legend_title_text="", title_x=0.2),

            fig.update_traces(
                mode='markers',
                marker=dict(size=15, color=eps['Marker_Color_Actual']))

            fig.update_traces(selector=dict(
                name='epsEstimate'), marker_color='white')
            fig.update_xaxes(mirror=True, ticks='outside', showline=True,
                             linecolor='black', showgrid=True, gridcolor='#292829', gridwidth=1,)
            fig.update_yaxes(mirror=True, ticks='outside', showline=True,
                             showgrid=True, gridcolor='#292829', gridwidth=1, linecolor='black',)

            graph = dcc.Graph(figure=fig, style={
                              'flex': '50%', 'margin': '10px'})
            row.append(graph)

        if len(row) == 3 or idx == len(selected_areas) - 1:
            plots.append(html.Div(row, style={'display': 'flex'}))
            row = []

    # print(sec_values)
    if len(sec_values) >= 1:
        for i in sec_values:
            if 'USD' in sec_d[i]['units'].keys():
                df = pd.DataFrame(sec_d[i]['units']['USD'])
                if timeline == 'quarterly':
                    df = df[df['form'] == '10-Q']
                else:
                    df = df[(df['form'] == '10-K') | (df['form'] == '10-K/A')]
                t = camel_case_to_title_case(i)
                # temp=pd.read_json(json.dumps(df_B[timeline])).transpose()
                fig = px.area(df, x='end', y='val', title=t,
                              width=450, height=400)
                text_color = '#FFFFFF'
                fig.update_layout(paper_bgcolor='#1C1C1C', plot_bgcolor='black', margin=dict(
                    r=10, l=0), title_text=t+' of '+ticker, title_x=0.5, title_font=dict(color=text_color), xaxis_title='Date',                  # Set title text color
                    # Set x-axis title and labels color
                    xaxis=dict(title_font=dict(color=text_color)),
                    # Set y-axis title and labels color
                    yaxis=dict(title_font=dict(color=text_color)),
                    # Set legend title and labels color
                    legend=dict(title_font=dict(color=text_color)),
                    font=dict(color=text_color))
                fig.update_xaxes(mirror=True, ticks='outside', showline=True,
                                 linecolor='black', showgrid=True, gridcolor='#292829', gridwidth=1,)
                fig.update_yaxes(mirror=True, ticks='outside', showline=True,
                                 showgrid=True, gridcolor='#292829', gridwidth=1, linecolor='black',)
                fig.update_traces(line=dict(color='#ffa028'))
                graph = dcc.Graph(figure=fig, style={
                                  'flex': '50%', 'margin': '10px'})
                row.append(graph)
            # if len(row) == 3 or idx == len(selected_areas) - 1:
                plots.append(html.Div(row, style={'display': 'flex'}))
                row = []
    return plots


def camel_case_to_title_case(input_string):
    words = []
    current_word = ""

    for char in input_string:
        if char.isupper() and current_word:
            words.append(current_word)
            current_word = char
        else:
            current_word += char

    if current_word:
        words.append(current_word)

    title_case_words = [word.capitalize() for word in words]
    return " ".join(title_case_words)


def fetch_data(url, header=None):
    res = requests.get(url, headers=header)
    return res.json()


def calculate_marker_color_actual(row):
    # print(row)
    if row['surprisePercent'] >= 0:
        return 'rgba(144, 238, 144, 1)'
    else:
        return 'rgba(240, 128, 128, 1)'


if __name__ == "__main__":
    app.run_server(debug=True)
