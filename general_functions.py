import pymap3d as pm
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import pandas as pd
import plotly.express as px
import pandas as pd


def map_plot(result, gw_ref):
    """
    _, _, _, gw_lat_lon = DataPreprocess().get_gw_cord_tdoa(row['gw_ref'], ds_json, gateway_locations, reference_position)

    # Creating a list of results for plotting in the map
    result = {
        'lat': [row['lat'], lat_est, row['pred_lat']] + [gw_lat_lon[i][0] for i in range(len(gw_lat_lon))],
        'lon': [row['lon'], lon_est, row['pred_lon']] + [gw_lat_lon[i][1] for i in range(len(gw_lat_lon))],
        'cat': ['Actual Pos', 'Estimated Pos', 'ML Predicted Pos'] + [f'GW Positions' for i in range(len(gw_lat_lon))]
    }
    """

    # Convert the data to a DataFrame
    df = pd.DataFrame(result)

    # Create a plot with Cartopy
    fig = plt.figure(figsize=(10, 5))
    ax = plt.axes(projection=ccrs.PlateCarree())

    # Add features to the map
    ax.add_feature(cfeature.COASTLINE)
    ax.add_feature(cfeature.BORDERS)
    ax.add_feature(cfeature.LAND)
    ax.add_feature(cfeature.OCEAN)

    # Define a colormap for the 'cat' categories
    categories = df['cat'].unique()
    colors = plt.cm.tab10(range(len(categories)))  # Use a colormap with enough distinct colors
    colormap = dict(zip(categories, colors))

    # Plot each category with its respective color
    for category in categories:
        subset = df[df['cat'] == category]
        ax.scatter(subset['lon'], subset['lat'], color=colormap[category], s=50, edgecolor='k', label=category,
                    transform=ccrs.PlateCarree())

    # Add a legend
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))

    # Add title and labels
    plt.title('Geographical Points Categorized by "cat"')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Save the plot to a file
    output_file = f'figs/map_plots/{gw_ref}.png'
    plt.savefig(output_file, bbox_inches='tight')
    # plt.show()

    print(f"Map saved as {output_file}")


def map_plot(df):
    color_scale = [(0, 'orange'), (1,'red')]

    fig = px.scatter_mapbox(df, 
                            lat='rxInfo_location_latitude', 
                            lon='rxInfo_location_longitude', 
                            color_continuous_scale=color_scale,
                            zoom=8, 
                            height=800,
                            width=800)

    fig.update_layout(mapbox_style="open-street-map")
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
    fig.show()