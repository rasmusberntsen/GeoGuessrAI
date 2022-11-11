# Imports
import plotly.express as px
import pandas as pd

# Creating a plot
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
    "count": [i for i in range(13)]
})
fig = px.choropleth(df, 
                locations="country", 
                color='count', 
                color_continuous_scale=px.colors.sequential.Plasma,
                locationmode='country names',
                scope="europe")
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
# Color denmark green
fig.update_traces(marker_line_color='black', marker_line_width=0.5, selector=dict(type='choropleth'))
# Remove other countries
fig.update_geos(fitbounds="locations", visible=False)
fig.show()