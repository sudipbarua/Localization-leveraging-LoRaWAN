from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer
import pandas as pd


def data_spliting(x_scaled, y, random_state, train_size):
    # Train, validation, test set splitting, (70%/15%/15%)
    x_train, x_test_val, y_train, y_test_val = train_test_split(x_scaled, y.values, train_size=train_size, random_state=random_state)
    x_val, x_test, y_val, y_test = train_test_split(x_test_val, y_test_val, test_size=0.5, random_state=random_state)
    print(f'Training shape:      {x_train.shape}')
    print(f'Test shape:          {x_test.shape}')
    print(f'Validation shape:    {x_val.shape}')

    return x_train, x_test, x_val, y_train, y_test, y_val 

def scaling(scaler, x):
    if scaler=='MinMax':
        min_max_scaler = MinMaxScaler()
        x_scaled = min_max_scaler.fit_transform(x)
        x_scaled_df = pd.DataFrame(x_scaled, columns=x.columns)
    elif scaler=='powed':
        pass
    elif scaler=='exponential':
        pass
    return x_scaled_df

def data_cleaning(ds, wathear_data=True):
    #### Remove entries with less than 3 gateways #### 
    columns = ds.columns
    # x = ds[columns[0:68]]  # Get basestations' RSS readings
    x = ds[columns[0:72]]  # Get basestations' RSS readings
    c = (x == -200).astype(int).sum(axis=1) # counting the amount of not-receiving gateways per message
    # c = 68 - c  # counting the amount of receiving gateways per message
    c = 72 - c  # counting the amount of receiving gateways per message
    c = c.tolist()

    # finding indices of messages with less than 3 receiving gateways, and dropping these messages from the dataset ds
    indices = list()
    for i in range(len(c)):
        element = c[i]
        if element <3:
            indices.append(i)  # appending all indices of messages with fewer than 3 receiving gateways

    print('Size before cleaning: ',ds.shape) # size before...
    ds = ds.drop(indices) # dropping all entries with fewer than 3 receiving gateways
    print('Size after cleaning: ',ds.shape) # ... and size after the dropping


    #### Dataset preparation for ML pipeline
    columns = ds.columns
    # x1 = ds[columns[0:68]] #features (RSS receptions)
    # x2 = ds[columns[69:71]] # SF and HDOP
    # x3 = ds[columns[75:]] #weather data  
    x1 = ds[columns[0:72]] #features (RSS receptions)
    x2 = ds[columns[73:75]] # SF and HDOP
    x3 = ds[columns[79:]] #weather data  
    if wathear_data==True:
        x = pd.concat([x1, x2, x3], axis=1)
    else:
        x = x1

    y = ds[columns[75:77]] # target (locations) 
    
    return x, y 

   
