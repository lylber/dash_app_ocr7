import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import numpy as np
import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import dash_bootstrap_components as dbc
import sys
sys.path.append(r'ressources')
from dico_features import *
from app import app

# Initialisation de l'application Dash
#app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

model = joblib.load(r'model\best_model.pkl')

scaler= model.named_steps['scaler']
data = pd.read_csv(r'datasets\train.csv').drop(['TARGET'],axis=1)

numerical_columns = data.columns[~data.columns.isin(colonnes_catégoriques)]


if 'r' in model.named_steps:
        rfe_step = model.named_steps['r']
    
        if hasattr(rfe_step, 'support_'):
            # If RFE step has a 'support_' attribute
            selected_features_mask = rfe_step.support_
            colonnes = data.columns[selected_features_mask]
        else:
            print("RFE step does not have 'support_' attribute.")
     
else:
    print("No 'r' (RFE) step found in the pipeline.")

# Layout pour l'application Dash
layout = html.Div(children=[ dbc.Container([
        dbc.Row([
            dbc.Col(html.H1("Prédition modèle", className="text-center",style={'fontFamily': 'Roboto'}), className="mb-0 mt-3")
        ]),
        html.Br(),
    *[dcc.Input(id=f'feature-{col}', type='number', placeholder=col) if col not in colonnes_catégoriques
      else dcc.Dropdown(id=f'feature-{col}', placeholder=col, options=[{'label': str(key), 'value': val} for key, val in
                                                       globals()[f'd_{col}'].items()], value=list(globals()[f'd_{col}'].values())[0] )#value=list(globals()[f'd_{col}'].keys())[0]
      for i, col in enumerate(data.columns)],
    html.Br(),
    html.Button('Prédiction', id='predict-button'),
    html.Div(id='prediction-output')
])
])

# Callback pour mettre à jour la sortie de la prédiction
@app.callback(
    Output('prediction-output', 'children'),
    [Input('predict-button', 'n_clicks')],
    [State(f'feature-{col}', 'value') for col in data.columns]
)
def update_prediction(n_clicks, *features):
    try:
        if n_clicks is not None:
            features_df = pd.DataFrame([features], columns=data.columns)
         
            scaled_features = pd.DataFrame(scaler.transform(features_df), columns=features_df.columns, index=features_df.index)
            print(scaled_features.isna().sum())
            prediction = model.predict(scaled_features)[0]
            probabilities = model.predict_proba(scaled_features)[0]
            positive_probability = probabilities[1]

            # Convertir la prédiction en texte significatif
            prediction_text = 'non solvable' if prediction == 1 else 'solvable'

            return (
                f"La prédiction du modèle est : {prediction_text}, "
                f"Probabilité : {positive_probability:.2f}"
            )
    except Exception as e:
        return f"Erreur : {str(e)}"

if __name__ == '__main__':
    app.layout = layout
    app.run_server(debug=True)
