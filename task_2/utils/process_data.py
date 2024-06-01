import pandas as pd
from sklearn.preprocessing import StandardScaler

def process_data(data, top_features=[6, 7, -1], train=True):
    copy = data.copy()
    data = copy.iloc[:, top_features]
    
    if train:
        features = data.drop(columns=['target'])
    else:
        features = data
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)


    data_processed = pd.DataFrame(features_scaled)
    if train:
        data_processed['target'] = data['target']
    return data_processed