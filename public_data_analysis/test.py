import seaborn as sns
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


ds = pd.read_csv('lorawan_dataset_antwerp.csv')
ds_mod = ds.copy()
ds_mod.columns = ds_mod.columns.str.replace("'", "")

fig = sns.pairplot(data=ds_mod, hue='SF')

fig.savefig('antwerp_DS_pairplot.png')

# Establish number of columns and rows needed to plot all features
n_cols = 5
n_elements = len(ds_mod.columns)
n_rows = np.ceil(n_elements / n_cols).astype("int")

# Specify y_value to spread data (ideally a continuous feature)
y_value = ds_mod['Latitude']  

# Create figure object with as many rows and columns as needed
fig, axes = plt.subplots(ncols=n_cols, nrows=n_rows, figsize=(15, n_rows * 2.5))

# Loop through features and put each subplot on a matplotlib axis object
for col, ax in zip(ds_mod.columns, axes.ravel()):
    sns.stripplot(data=ds_mod, x=col, y=y_value, ax=ax, palette="tab10", size=1, alpha=0.5)
plt.tight_layout()

plt.savefig('antwerp_ds_stirplot.png')
