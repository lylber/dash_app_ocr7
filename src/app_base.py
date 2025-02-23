import dash
import dash_bootstrap_components as dbc

# bootstrap theme
# https://bootswatch.com/lux/
external_stylesheets = [dbc.themes.LUX]
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, assets_folder = 'assets', assets_url_path = '/assets')

app.config.suppress_callback_exceptions = True

# https://app-dash-opc7.onrender.com 