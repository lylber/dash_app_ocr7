import os
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

from app import app
import globall  
import home 
import prediction 
import API 
# Définition des couleurs
colors = {
    'background': '#B0C4DE',
    'text': '#00152F'}
server = app.server

# Barre de navigation
navbar = dbc.Navbar(
    dbc.Container(
        [
            dbc.NavbarBrand("Prêt à dépenser", href="/home", className="ms-2", style={'color': "red"}),  # Changement ici
            dbc.Nav(
                [
                    dbc.NavLink("Home", href="/home"),
                    dbc.NavLink("Prediction base client", href="/prediction"),
                    dbc.NavLink("Statistiques générales", href="/global"),
                    dbc.NavLink("Prediction API", href="/api"),
                ],
                className="mr-auto",
            ),
        ],
    ),
    color="darkblue",
    dark=True,
    className="mb-4",
)

# Fonction pour gérer l'ouverture/fermeture de la barre de navigation
def toggle_navbar_collapse(n, is_open):
    if n:
        return not is_open
    return is_open

# Configuration des callbacks pour le toggle de la barre de navigation
for i in [2]:
    app.callback(
        Output(f"navbar-collapse{i}", "is_open"),
        [Input(f"navbar-toggler{i}", "n_clicks")],
        [State(f"navbar-collapse{i}", "is_open")],
    )(toggle_navbar_collapse)

# Mise en page de l'application
app.layout = html.Div([
    dcc.Location(id='url', refresh=False),
    navbar,
    html.Div(id='page-content')
])

# Callback pour afficher le contenu de la page en fonction de l'URL
@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')], allow_duplicate=True)
def display_page(pathname):
    try:
        print(f"Displaying page for pathname: {pathname}")

        if pathname == '/prediction':
            return prediction.layout
        elif pathname == '/global':
            return globall.layout
        elif pathname == '/api':
            return API.layout
        else:
            return home.layout

    except Exception as e:
        print(f"Error: {str(e)}")
        # Vous pouvez également logger l'erreur dans un fichier de logs si nécessaire
        return html.Div(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run_server(host='127.0.0.1', debug=True)
