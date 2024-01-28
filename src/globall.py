import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import dash_bootstrap_components as dbc
import plotly.express as px
from app_base import app

# Chargement des données
data = pd.read_csv(r'datasets/brut_test.csv')
numeric_columns = data.select_dtypes(include=['number']).columns

# Parcourez les colonnes numériques et appliquez la transformation
for col in numeric_columns:
    if (data[col] < 0).all():
        data[col] = data[col].abs()
# Paramètres de style
param = {'color': '#8badda'}
colors = {
    'background': '#B0C4DE',
    'text': '#00152F'}

# Initialisation de l'application Dash
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Mise en page de l'application
layout = html.Div(children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Statistiques", className="text-center",style={'fontFamily': 'Roboto'}), className="mb-0 mt-3")
        ]),
        html.Br(),
        dbc.Row([
            dbc.Col(html.Label("Variables client :", style={'fontWeight': 'bold'}), width=2),
            dbc.Col(dcc.Dropdown(
                id='paramètres_id',
                options=[{'label': str(i), 'value': i} for i in data.columns],
                value=[data.columns[3], data.columns[4], data.columns[5], data.columns[9]],
                multi=True,
                style={
                    'backgroundColor': param['color'],
                    'fontWeight': 'bold'  # Mettre en gras le texte
                }
            ), width=4),
            dbc.Col(html.Label("Entrez l'ID du client :", style={'fontWeight': 'bold'}), width=2),
            dbc.Col(dcc.Dropdown(
                id='input-customer-id',
                options=[{'label': str(i), 'value': i} for i in data['SK_ID_CURR']],
                value=None,
                style={
                    'backgroundColor': param['color'],
                    'fontWeight': 'bold'  # Mettre en gras le texte
                }
            ), width=4),
        ], className="form-inline"),
        html.Br(),
        dbc.Row([
            
            dbc.Col(dcc.Graph(id='prediction-chart')),
            
            dbc.Col(dcc.Graph(id='prediction-chart2')),
        ],style = {'width': '100%', 'display': 'flex', 
                        'align-items': 'center', 'justify-content': 'center'}),
        html.Br(),
        dbc.Row([
            
            dbc.Col(dcc.Graph(id='prediction-chart3')),
            
            dbc.Col(dcc.Graph(id='prediction-chart4')),
        ]),
    ])
])

# Callback pour mettre à jour la sortie en fonction du paramètre et de l'ID du client sélectionnés
@app.callback(
    [Output('prediction-chart', 'figure'),  
     Output('prediction-chart2', 'figure'),   
     Output('prediction-chart3', 'figure'),
     Output('prediction-chart4', 'figure')],
    [Input('paramètres_id', 'value'),
     Input('input-customer-id', 'value')]
)

def plot_column_data(param_ids, input_customer_id=None):
    figures = []

    for i, param_id in enumerate(param_ids):
        if data[param_id].dtype == 'float':
            fig = px.histogram(data[param_id], nbins=30,x=param_id,)
            fig.update_layout(
                        margin=dict(l=20, r=20, t=20, b=20), showlegend=False)
            if input_customer_id is not None:
                valeur_affichee = data[data['SK_ID_CURR'] == input_customer_id][param_id].values[0]
                fig.add_shape(
                    type='line',
                    x0=valeur_affichee,
                    x1=valeur_affichee,
                    y0=0,
                    y1=1,
                    line=dict(color='red', width=2, dash='dash')
                )
                
                # Find the bin index corresponding to valeur_affichee
                bin_index = next(
                    (i for i, bin_edge in enumerate(fig.data[0].x[:-1]) if bin_edge <= valeur_affichee < fig.data[0].x[i + 1]),
                    len(fig.data[0].x) - 1
                )

                # Update the color of the selected bin to red
                fig.update_traces(marker_color=['red' if i == bin_index else 'blue' for i in range(len(fig.data[0].x) - 1)])

                # Add black marker lines for visibility
                fig.update_traces(marker_line_color='black', marker_line_width=1, selector=dict(type='bar'))

                # Update layout to add black borders to bars and hide legend
                fig.update_layout(bargap=0.1, bargroupgap=0.1, showlegend=False)

                # Add annotation for the exact value of valeur_affichee at the center of the bin
                bin_center = (fig.data[0].x[bin_index] + fig.data[0].x[bin_index + 1]) / 2
                bin_height = fig.data[0].y[bin_index] if fig.data[0].y is not None else 0

                fig.add_annotation(
                    x=bin_center,
                    y=bin_height,
                    text=str(valeur_affichee),
                    showarrow=True,
                    arrowhead=2,
                    arrowcolor='black',
                    arrowwidth=2,
                    font=dict(color='black')
                )

            figures.append(fig)
        elif data[param_id].dtype == 'int64':
            value_counts = data[param_id].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values)
            fig.update_layout(
                        margin=dict(l=20, r=20, t=20, b=20))
            if input_customer_id is not None:
                valeur_affichee = data[data['SK_ID_CURR'] == input_customer_id][param_id].values[0]
                fig.add_shape(
                    type='line',
                    x0=valeur_affichee,
                    x1=valeur_affichee,
                    y0=0,
                    y1=value_counts.max(),
                    line=dict(color='red', width=2, dash='dash')
                )
            figures.append(fig)
        elif data[param_id].dtype == 'object':
            value_counts = data[param_id].value_counts()
            fig = px.bar(x=value_counts.index, y=value_counts.values)
            fig.update_layout(
                        margin=dict(l=20, r=20, t=20, b=20))
            if input_customer_id is not None:
                valeur_affichee = data[data['SK_ID_CURR'] == input_customer_id][param_id].values[0]
                fig.add_shape(
                    type='line',
                    x0=str(value_counts.index[0]),
                    x1=str(value_counts.index[0]),
                    y0=0,
                    y1=value_counts.max(),
                    line=dict(color='red', width=2, dash='dash')
                )
            figures.append(fig)
        else:
            print(f"Type de données non pris en charge pour {param_id}")

    return figures[0], figures[1], figures[2], figures[3]


if __name__ == '__main__':
    app.run_server(debug=True)
