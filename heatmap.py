# Imports
import random
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Creating a plot
ground_truth = 7
df = pd.DataFrame({
    "country": [
        "Germany",
        "Denmark",
        "Estonia",
        "Spain",
        "France",
        "United Kingdom",
        "Greece",
        "Italy",
        "Norway",
        "Poland",
        "Romania",
        "Sweden",
        "Ukraine",
    ],
    "Propability": [random.random()/10 for i in range(13)]#[i/12 for i in range(13)]

})
df["border_color"] = "black"
df["border_width"] = 1
df["border_color"].iloc[ground_truth] = "#EB4144"
df["border_width"].iloc[ground_truth] = 2
fig = px.choropleth(df,
                    locations="country",
                    color='Propability',
                    # Color scale from red to green
                    color_continuous_scale=px.colors.sequential.Blues,
                    locationmode='country names',
                    scope="europe",

                    )
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
# Color denmark green
fig.update_traces(marker_line_color=df["border_color"],
                  marker_line_width=df["border_width"],
                  selector=dict(type='choropleth'),)
fig.update_geos(
    # fitbounds="locations",
    visible=False)

# Zoom in on denmark
fig.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=600,
    width=600,
    showlegend=False,
    paper_bgcolor='rgba(255,255,255,1)',
    plot_bgcolor='rgba(0,0,0,0)',
)

# Zoom out
fig.update_layout(
    margin={"r": 0, "t": 0, "l": 0, "b": 0},
    height=600,
    width=600,
    showlegend=False,
    paper_bgcolor='rgba(255,255,255,1)',
    plot_bgcolor='rgba(0,0,0,0)',
    geo=dict(
        projection_scale=1.8,
        center=dict(lat=55.25, lon=9.5018),
        resolution=50,
    )
)


fig.show()