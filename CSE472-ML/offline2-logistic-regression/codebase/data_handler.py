import pandas as pd


def load_dataset():
    dataset = pd.read_csv("./data_banknote_authentication.csv")
    
    X = dataset.drop(columns=["isoriginal"])
    y = dataset["isoriginal"]
    return X, y


def split_dataset(X, y, test_size=0.2, shuffle=True):
    dataset = pd.DataFrame(X)
    dataset["isoriginal"] = y

    if shuffle:
        dataset = dataset.sample(frac=1.0)

    test_data = dataset.sample(frac=test_size).reset_index(drop=True)
    train_data = dataset.drop(test_data.index).reset_index(drop=True)
    
    X_test = test_data.drop(columns=["isoriginal"])
    y_test = test_data["isoriginal"].to_numpy()

    X_train = train_data.drop(columns=["isoriginal"])
    y_train = train_data["isoriginal"].to_numpy()

    return X_train, y_train, X_test, y_test


def bagging_sampler(X, y):
    dataset = pd.DataFrame(X)
    dataset["isoriginal"] = y
    
    # Dont set random_state here, because we dont want repition
    dataset = dataset.sample(frac=1.0, replace=True)

    new_X = dataset.drop(columns=["isoriginal"])
    new_y = dataset["isoriginal"].to_numpy()

    return new_X, new_y