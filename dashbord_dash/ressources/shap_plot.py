import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.exceptions import InconsistentVersionWarning
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import dash_table
import warnings
import shap


# add blocks for risk groups
bot_val = 0.5
top_val = 1


# Define the get_heat_color function to generate a gradient color
def get_heat_color(value):
    # Ensure the value is within the valid range [0, 1]
    value = max(0, min(1, value))
    
    # Calculate color based on the importance value
    red = int((1 - value) * 255)
    green = int(value * 255)
    return f'rgb({red}, {green}, 0)'



def shap_plot(input_features_scaled, positive_probability,logistic_regression_model,train,selected_columns):


    if positive_probability/100 <= 0.275685:
        risk_grp = 'peu de risque'
    elif positive_probability/100 <= 0.795583:
        risk_grp = 'risque moyen'
    else:
        risk_grp = 'fort risque'

    color = get_heat_color(positive_probability)

    # create a single bar plot showing likelihood of heart disease
    fig1 = go.Figure()
    fig1.add_trace(go.Bar(
        y=[''],
        x=[(positive_probability*100).round(2)],
        marker_color=color,
        orientation='h',
        width=1,
        text=' probabilité de non solvable',
        textposition='auto',
        hoverinfo='skip'
    ))

    fig1.add_shape(
        type="rect",
        x0=0,
        y0=bot_val,
        x1=0.275686 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="green"
    )
    fig1.add_shape(
        type="rect",
        x0=0.275686 * 100,
        y0=bot_val,
        x1=0.795584 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="orange"
    )
    fig1.add_shape(
        type="rect",
        x0=0.795584 * 100,
        y0=bot_val,
        x1=1 * 100,
        y1=top_val,
        line=dict(
            color="white",
        ),
        fillcolor="red"
    )
    fig1.add_annotation(
        x=0.275686 / 2 * 100,
        y=0.75,
        text="Risque Faible",
        showarrow=False,
        font=dict(color="black", size=14)
    )
    fig1.add_annotation(
        x=0.53 * 100,
        y=0.75,
        text="Risque Moyen",
        showarrow=False,
        font=dict(color="black", size=14)
    )
    fig1.add_annotation(
        x=0.9 * 100,
        y=0.75,
        text="Risque Fort",
        showarrow=False,
        font=dict(color="black", size=14)
    )
    fig1.update_layout(margin=dict(l=0, r=50, t=10, b=10), xaxis={'range': [0, 100]})

    # do shap value calculations for basic waterfall plot
    explainer_patient = shap.Explainer(logistic_regression_model, train, feature_perturbation="interventional")
    shap_values_patient = explainer_patient.shap_values(input_features_scaled[selected_columns])
    updated_fnames = input_features_scaled.T.reset_index()
    updated_fnames.columns = ['feature', 'value']
    updated_fnames['shap_original'] = pd.Series(shap_values_patient[0])
    updated_fnames['shap_abs'] = updated_fnames['shap_original'].abs()
    updated_fnames = updated_fnames.sort_values(by=['shap_abs'], ascending=True)

    # need to collapse those after first 9, so plot always shows 10 bars
    show_features = 9
    num_other_features = updated_fnames.shape[0] - show_features
    col_other_name = f"{num_other_features} other features"
    f_group = pd.DataFrame(updated_fnames.head(num_other_features).sum()).T
    f_group['feature'] = col_other_name
    plot_data = pd.concat([f_group, updated_fnames.tail(show_features)])

    # additional things for plotting
    plot_range = plot_data['shap_original'].cumsum().max() - plot_data['shap_original'].cumsum().min()
    plot_data['text_pos'] = np.where(plot_data['shap_original'].abs() > (1/9)*plot_range, "inside", "outside")
    plot_data['text_col'] = "white"
    plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] < 0), 'text_col'] = "#3283FE"
    plot_data.loc[(plot_data['text_pos'] == "outside") & (plot_data['shap_original'] > 0), 'text_col'] = "#F6222E"

    fig2 = go.Figure(go.Waterfall(
        name="",
        orientation="h",
        measure=['absolute'] + ['relative']*show_features,
        base=explainer_patient.expected_value,
        textposition=plot_data['text_pos'],
        text=plot_data['shap_original'],
        textfont={"color": plot_data['text_col']},
        texttemplate='%{text:+.2f}',
        y=plot_data['feature'],
        x=plot_data['shap_original'],
        connector={"mode": "spanning", "line": {"width": 1, "color": "rgb(102, 102, 102)", "dash": "dot"}},
        decreasing={"marker": {"color": "#3283FE"}},
        increasing={"marker": {"color": "#F6222E"}},
        hoverinfo="skip"
    ))
    fig2.update_layout(
        waterfallgap=0.2,
        autosize=False,
        width=800,
        height=400,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        yaxis=dict(
            showgrid=True,
            zeroline=True,
            showline=True,
            gridcolor='lightgray'
        ),
        xaxis=dict(
            showgrid=False,
            zeroline=False,
            showline=True,
            showticklabels=True,
            linecolor='black',
            tickcolor='black',
            ticks='outside',
            ticklen=5
        ),
        margin={'t': 25, 'b': 50},
        shapes=[
            dict(
                type='line',
                yref='paper', y0=0, y1=1.02,
                xref='x', x0=plot_data['shap_original'].sum()+explainer_patient.expected_value,
                x1=plot_data['shap_original'].sum()+explainer_patient.expected_value,
                layer="below",
                line=dict(
                    color="white",
                    width=1,
                    dash="dot")
            )
        ]
    )
    fig2.update_yaxes(automargin=True)
    fig2.add_annotation(
        yref='paper',
        xref='x',
        x=explainer_patient.expected_value,
        y=-0.12,
        text="E[f(x)] = {:.2f}".format(explainer_patient.expected_value),
        showarrow=False,
        font=dict(color="blue", size=14)
    )

    fig2.add_annotation(
        yref='paper',
        xref='x',
        x=plot_data['shap_original'].sum()+explainer_patient.expected_value,
        y=1.075,
        text="f(x) = {:.2f}".format(plot_data['shap_original'].sum()+explainer_patient.expected_value),
        showarrow=False,
        font=dict(color="black", size=14)
    )

    return fig1, fig2