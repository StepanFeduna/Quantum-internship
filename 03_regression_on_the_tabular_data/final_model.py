import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import Ridge

BEST_FEATURES = None


def features_labels_split(df):
    X = df.drop("target", axis=1)
    y = df["target"].copy()

    return X, y


def best_features_search(X, y):
    forest_reg = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    forest_reg.fit(X, y)
    global BEST_FEATURES
    BEST_FEATURES = forest_reg.feature_importances_ > 1e-8


def data_pipeline(X):
    preprocessing = make_pipeline(
        PolynomialFeatures(degree=2, include_bias=False), StandardScaler()
    )

    X_prepared = preprocessing.fit_transform(X)

    return X_prepared


def train_model(df):
    X, y = features_labels_split(df)

    best_features_search(X, y)

    X = X.iloc[:, BEST_FEATURES]

    X_prepared = data_pipeline(X)

    ridge_reg = Ridge(alpha=1e-6)
    ridge_reg.fit(X_prepared, y)

    return ridge_reg


def make_prediction(X, model):
    X = X.drop(["target"], axis=1, errors="ignore")
    X = X.iloc[:, BEST_FEATURES]

    X_prepared = data_pipeline(X)
    y_predict = model.predict(X_prepared)

    return y_predict


def main():
    model = None

    while True:
        print(
            "(q-Exit) Input:\n\ttrain /dataset.csv to train model\n\tpredict /dataset.csv to make a prediction.\n"
        )
        console_input = input()

        if console_input == "q":
            break

        if console_input[-4:] != ".csv":
            print("Dataset should be in .csv format.\n")
            continue

        func, df_path = console_input.split(sep=" ", maxsplit=1)

        if func == "train":
            train_df = pd.read_csv(df_path)

            if "target" not in train_df.columns:
                print("Train dataset should contain 'target' labels.\n")
                continue

            print("Training will take around 30 seconds.\n")

            model = train_model(train_df)

            print("Model sucsesfuly trained.\n")

        elif func == "predict":
            if model is None:
                print("Please train the model first.\n")
                continue

            test_df = pd.read_csv(df_path)

            target_predict = make_prediction(test_df, model)

            target_predict = pd.DataFrame(target_predict, columns=["target_predict"])
            target_predict.to_csv("model_predictions.csv")

            print("Prediction saved to model_predictions.csv\n")

        else:
            print("Wrong command\n")
            continue


if __name__ == "__main__":
    main()
