import dash
import dash_bootstrap_components as dbc

# bootstrap theme
# https://bootswatch.com/lux/
external_stylesheets = [dbc.themes.LUX]#.LUX
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, assets_folder = 'assets', assets_url_path = '/assets')
#server = app.server

app.config.suppress_callback_exceptions = True

#server = app.server