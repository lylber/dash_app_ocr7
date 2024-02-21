import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import InconsistentVersionWarning
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash_table
import warnings
import sys
sys.path.append(r'ressources/')
import base64
from ressources.shap_plot import *
from app_base import app

# Suppress the InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# telecharger et traiter les datasets train/test
def load_data():
    
    model = joblib.load(r'model/best_model.pkl')
    data = pd.read_csv(r'datasets/test.csv')
    data_brut=pd.read_csv(r'datasets/brut_test.csv')
    train=pd.read_csv(r'datasets/train.csv')

    scaler= model.named_steps['scaler']

    data_brut=data_brut[data_brut['SK_ID_CURR'].isin(data.SK_ID_CURR)]
    d = data.drop(['SK_ID_CURR'], axis=1)
    scaled_features = scaler.transform(d)
    df_scaled = pd.DataFrame(scaled_features, columns=d.columns)
    df_scaled['SK_ID_CURR'] = data['SK_ID_CURR'].astype(int)
    data_brut['AGE']=data['AGE']
    data_brut['AGE']=data_brut['AGE'].astype(int)

    if 'r' in model.named_steps:
        rfe_step = model.named_steps['r']
    
        if hasattr(rfe_step, 'support_'):
            # If RFE step has a 'support_' attribute
            selected_features_mask = rfe_step.support_
            selected_columns = data.drop(['SK_ID_CURR'],axis=1).columns[selected_features_mask]
        else:
            print("RFE step does not have 'support_' attribute.")
     
    else:
        print("No 'r' (RFE) step found in the pipeline.")

    scaler= model.named_steps['scaler']

    return model, df_scaled, data_brut, selected_columns, scaler, train

model, data, data_brut, selected_columns, scaler, train = load_data()

train=pd.DataFrame(scaler.transform(train.drop(['TARGET'],axis=1)), columns=train.drop(['TARGET'],axis=1).columns, index=train.index)
train=train[selected_columns]

logistic_regression_model = model.named_steps['m']



# Initialisation de l'application Dash
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

param = {'color': '#8badda'}

image_filename = r'ressources/shap.plots.beeswarm(explainer(train)).png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())


# Create a sample layout for the DataTable
table_layout = dash_table.DataTable(
    id='data-table',
    columns=[
        {'name': 'SK_ID_CURR', 'id': 'SK_ID_CURR'},
        {'name': 'CODE_GENDER', 'id': 'CODE_GENDER'},
        {'name': 'AGE', 'id': 'AGE'},
        {'name': 'NAME_FAMILY_STATUS', 'id': 'NAME_FAMILY_STATUS'},
        {'name': 'CNT_CHILDREN', 'id': 'CNT_CHILDREN'}

    ],
    style_table={'height': '300px', 'overflowY': 'auto'}
)

#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

layout = html.Div(children=[
    dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Prédiction du modèle", className="text-center",style={'fontFamily': 'Roboto'}), className="mb-0 mt-3")
        ]),
        html.Label("Entrez l'ID du client :"),
        dcc.Dropdown(
            id='input-customer-id',
            options=[{'label': str(i), 'value': i} for i in data['SK_ID_CURR']],
            value=data['SK_ID_CURR'].iloc[0],  # Set the default value to the first item in the list
            style={
                'backgroundColor': param['color'],
                'fontWeight': 'bold',
                'width':'150px'
            }
        ),
        html.Br(),

        html.Div(id='output-prediction'),
        html.Br(),
        html.Div(
        dcc.Graph(id='probability-progress-bar'),
        style={'display': 'flex', 'justify-content': 'center'}
        ),
        html.Br(),
        dbc.Row([
 
        
        dbc.Col(html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()), style={'width': '100%', 'height': '400px',}), width=6, className='my-5', style={'marginTop': '50px','marginLeft': 'auto'}),
        dbc.Col(dcc.Graph(id='prediction-chart1'), width=6, className='my-5', style={'marginTop': '0px','width': '48%','marginRight': '25px'}),
        ], style={'width': '100%', 'display': 'flex', 'align-items': 'center', 'justify-content': 'center'}),


     #   dcc.Graph(id='prediction-chart1'),
        html.Br(),
        table_layout
    ])
])

# ...

# ...

@app.callback(
    [Output('output-prediction', 'children'),
     Output('prediction-chart1', 'figure'),
     Output('probability-progress-bar', 'figure'),
     Output('data-table', 'data')],
    [Input('input-customer-id', 'value')],
    allow_duplicate=True
)

def update_prediction(customer_id,threshold=0.6):
    
    try:
        customer_id = int(customer_id)
        
        input_features = data[data['SK_ID_CURR'] == customer_id].loc[:, data.columns != 'SK_ID_CURR']

        input_features_scaled = pd.DataFrame(scaler.transform(input_features), columns=input_features.columns, index=input_features.index)

    #    prediction = model.predict(input_features_scaled)[0] #prediction sans seuil
        probabilities = model.predict_proba(input_features_scaled)[0]
        positive_probability = round(probabilities[1],2)
        y_pred_thresholded = (positive_probability >= threshold).astype(int)        
        print(f"DEBUG: Prediction: {y_pred_thresholded}, Positive Probability: {positive_probability}")

        plot=shap_plot(input_features_scaled,positive_probability,logistic_regression_model, train,selected_columns)
        
        fig=plot[1]
        # Add a horizontal bar for each variable with a gradient color     
        if y_pred_thresholded == 1:
            prediction='non solvable'
        else:
            prediction='solvable'
            if positive_probability < 0.5:
                positive_probability=round(probabilities[0],2)
            else:
                positive_probability=round(probabilities[0],2)
                positive_probability=f'{positive_probability} seuil de solvabilité dépassé'
        # Update the progress bar value based on the positive probability
        progress_value = plot[0]
        # Fetch AGE and NAME_FAMILY_STATUS for the selected SK_ID_CURR
        selected_data = data_brut.loc[data_brut['SK_ID_CURR'] == customer_id,  ['SK_ID_CURR', 'AGE','CODE_GENDER', 'NAME_FAMILY_STATUS', 'CNT_CHILDREN',]]
        table_data = selected_data.to_dict(orient='records')
        print(f"DEBUG: Table Data: {table_data}")

        return (
            f"La prédiction du modèle pour le client {customer_id} est : {prediction}, "
            f" probabilité  : {positive_probability}",
            fig,
            progress_value,
            table_data
        )
    except Exception as e:
        print(f"ERROR: {e}")
        return f"Erreur : ceci ne correspond pas à un numéro client connu", None, 0, {"height": "20px"}, []

# ...



if __name__ == '__main__':
    app.run_server(debug=True)
