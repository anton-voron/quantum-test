import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib

from .utils import process_data





def get_data():
    top_features = [6, 7, -1]
    train_path = '../train.csv'

    data = pd.read_csv(train_path)
    return process_data(data, top_features)


def save_model(model):
    model_filename = f"random_forest_model_{pd.Timestamp.now().strftime('%d_%m_%Y_%H_%M')}.joblib"
    joblib.dump(model, model_filename)
    print(f"Model saved as {model_filename}")


def main():
    data = get_data()

    X = data.drop(columns=['target'])
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    model = RandomForestRegressor(n_estimators=400, min_samples_split=2, min_samples_leaf=1, max_depth=30, bootstrap=True, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    print(f"Random Forest RMSE: {rmse}")

    save_model(model)


if __name__ == '__main__':
    main()