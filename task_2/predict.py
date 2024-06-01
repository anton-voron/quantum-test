import pandas as pd
import joblib

from utils import process_data


def get_data():
    top_features = [6, 7]
    file_name = '../hidden_test.csv'

    data = pd.read_csv(file_name)
    return process_data(data, top_features, train=False)


def load_model(file_path):
    return joblib.load(file_path)


def save_predictions(predictions):
    predictions.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")


def main():
    model_filename = './random_forest_model_01_06_2024_22_39.joblib'
    data = get_data()

    X = data
    model = joblib.load(model_filename)

    y_pred = model.predict(X)
    print(y_pred)

    y_pred = pd.DataFrame(y_pred, columns=['target'])
    y_pred.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")


main()    


if __name__ == '__main__':
    main()