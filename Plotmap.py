import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def plot_map(probs, ground_truth, countries):
    world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
    plot_countries = world[world["name"].isin(countries)]
    plot_countries = plot_countries[["name", "geometry"]].reset_index(drop=True)
    plot_countries["prob"] = 0
    plot_countries["prob"] = plot_countries["name"].map(probs)
    # getting the prediction as the country with the highest probability
    prediction = plot_countries["name"].iloc[plot_countries["prob"].idxmax()]
    print(prediction)
    ax = plot_countries.plot(column="prob", figsize=(15, 10), edgecolor="black")
    plot_countries[plot_countries["name"] == ground_truth].plot(
        ax=ax, facecolor="none", edgecolor="lime", linewidth=4
    )
    plot_countries[plot_countries["name"] == prediction].plot(
        ax=ax, facecolor="none", edgecolor="red", linewidth=1
    )
    # add colorbar and remove axis and frame
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=1))
    ax.axis("off")  # remove axis
    plt.colorbar(sm)
    plt.title("Predictions")
    # legend with "green": Ground truth
    green_patch = mpatches.Patch(color="lime", label="Ground truth")
    # legend with "red": Prediction
    red_patch = mpatches.Patch(color="red", label="Prediction")
    plt.legend(handles=[green_patch, red_patch], loc="center left")
    plt.show()


p = 1 / 13
probs = {
    "Germany": p,
    "Denmark": p,
    "Estonia": p,
    "Spain": p,
    "France": p,
    "United Kingdom": p,
    "Greece": 1,
    "Italy": p,
    "Norway": p,
    "Poland": p,
    "Romania": p,
    "Sweden": p,
    "Ukraine": p,
}

countries = [
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
]
ground_truth = "Greece"
plot_map(probs, ground_truth, countries)
