import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import os
import base64
import sys
sys.path.append(r'ressources')
from dico_features import data_dictionary

# Récupérer le chemin du script
# Charger l'image et la convertir en base64
with open(os.path.join( "ressources", "logo_projet_fintech.png"), "rb") as image_file:
    encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

from app_base import app

# Initialiser l'application Dash
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# Définir la mise en page
layout = html.Div(children=[
    dbc.Container([
        # Première ligne
        dbc.Row([
            dbc.Col(html.H1("Dashboard clients", className="text-center", style={'fontFamily': 'Roboto'}), className="mb-0 mt-3")
        ]),
        html.Br(),

        # Troisième ligne avec l'image centrée
        dbc.Row([
            dbc.Col(html.Img(src=f'data:image/png;base64,{encoded_image}'), style={'textAlign': 'center'})
        ]),

        html.Br(),
        html.Br(),

        # Quatrième ligne avec le titre
        html.H4("Glossaire paramètres client ", style={'fontFamily': 'Roboto'}),

        # Dropdown pour sélectionner un mot
        dcc.Dropdown(
            id='mot-dropdown',
            options=[{'label': mot, 'value': mot} for mot in data_dictionary.keys()],
            value=list(data_dictionary.keys())[0],  # Valeur par défaut
            style={'width': '50%', 'backgroundColor': '#8badda', 'fontWeight': 'bold', 'fontFamily': 'Roboto'}  # Appliquer la police ici
        ),     
        html.Br(),

        # Div pour afficher la signification
        html.Div(id='signification-output', style={'fontFamily': 'Roboto'}),
    ])
])

# Callback pour mettre à jour la sortie en fonction du mot sélectionné
@app.callback(
    Output('signification-output', 'children'),
    [Input('mot-dropdown', 'value')]
)
def update_output(selected_mot):
    signification = data_dictionary.get(selected_mot, "Aucune signification trouvée")
    return html.Div([
        html.H6(f"Signification du paramètre '{selected_mot}': {signification}", style={'fontFamily': 'Roboto'}),
    ])

# Exécuter l'application
if __name__ == '__main__':
    app.run_server(debug=True)
