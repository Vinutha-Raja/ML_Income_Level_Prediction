from sklearn import preprocessing
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
import pandas as pd
import numpy as np


attr_numerical = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",]


def read_data(file_name):
    data = []
    with open(file_name, 'r') as f:
        for line in f:
            data.append(line.strip().split(','))

    dataset = []
    for r in data:
        f = []
        for x in r:
            if x.isnumeric():
                f.append(float(x))
            else:
                f.append(x)
        dataset.append(f)
    return dataset


def read_df_from_csv(file_name, test=False):
    if not test:
        data_df = pd.read_csv(file_name, header=None, names=["age", "workclass", "fnlwgt", "education", "education-num",
                                                         "marital-status", "occupation", "relationship",
                                                         "race", "sex", "capital-gain", "capital-loss",
                                                         "hours-per-week", "native-country", "label"])
    else:
        data_df = pd.read_csv(file_name, header=None, names=["ID", "age", "workclass", "fnlwgt", "education", "education-num",
                                                             "marital-status", "occupation", "relationship",
                                                             "race", "sex", "capital-gain", "capital-loss",
                                                             "hours-per-week", "native-country"])

    return data_df


def convert_numerical_columns(data_df):
    # print(data_df)
    for i in attr_numerical:
        # print(i)
        m = data_df[i].median()
        data_df[i] = data_df[i].apply(lambda x: 1 if x > m else 0)
    return data_df


def encode_categorical_data(data):
    categorical_attr = ["workclass", "education", "marital-status", "occupation", "relationship",
                                                             "race", "sex", "native-country"]
    for attr in categorical_attr:
        le = preprocessing.LabelEncoder()
        le.fit(data[attr].tolist())
        data[attr] = le.transform(data[attr].tolist())
    # print(data)
    return data


def one_hot_encoding(data):
    categorical_attr = ["workclass", "education", "marital-status", "occupation", "relationship",
                        "race", "sex", "native-country"]
    ohe = preprocessing.OneHotEncoder(sparse=False, dtype=int)
    feature_array = ohe.fit_transform(data[categorical_attr])
    # print("feature_array", feature_array)
    # print("ohe.categories_", ohe.categories_)
    feature_labels = ohe.categories_
    # print(type(feature_labels))
    feature_list = []
    for ele in feature_labels:
        for item in ele:
            feature_list.append(item)
    print(feature_list)
    print(pd.DataFrame(feature_array, columns=feature_list))
    encoded_df = pd.DataFrame(feature_array, columns=feature_list)
    # print("data_shape", data.shape, "encoded_df", encoded_df.shape)
    data = data.drop(columns=categorical_attr)
    data_new = pd.concat([data, encoded_df], axis=1)

    # print("data_new", data_new, data_new.shape)
    return data_new


def one_hot_encoding1(data, categorical_index, encoder):
    cat_data = data[:, categorical_index]
    num_data = np.delete(data, categorical_index, axis=1)
    cat_data_enc = encoder.transform(cat_data)
    encoded_data = np.concatenate((num_data, cat_data_enc), axis=1)
    return encoded_data


def get_catboost_prediction():
    train = read_df_from_csv("./income2022f/train_final1.csv")
    test = read_df_from_csv("./income2022f/test_final1.csv", True)

    X_train = train.iloc[:, 0:-1].to_numpy()
    labels = train.iloc[:, -1].to_numpy()
    X_test = test.iloc[:, 1:].to_numpy()

    categorical_index = [1, 3, 5, 6, 7, 8, 9, 13]
    enc = preprocessing.OneHotEncoder(sparse=False, dtype=int)
    enc.fit(np.concatenate((X_train[:, categorical_index], X_test[:, categorical_index]), axis=0))

    encoded_train = one_hot_encoding1(X_train, categorical_index, enc)
    # encoded_train['Holand-Netherlands'] = 0
    print("encoded_train")
    print(encoded_train)

    id_list = test['ID'].tolist()
    test = test.drop(columns=['ID'])
    encoded_test = one_hot_encoding1(X_test, categorical_index, enc)
    print("encoded_test")
    print(encoded_test)

    scaler = preprocessing.StandardScaler()
    encoded_train1 = scaler.fit_transform(encoded_train)
    encoded_test1 = scaler.transform(encoded_test)

    # Splitting training data and calculating accuracy

    X_train, X_test, Y_train, Y_test = train_test_split(
        encoded_train1, labels, test_size=0.2, random_state=42)

    model = CatBoostClassifier()
    # Train accuracy
    model.fit(X_train, Y_train)
    train_pred = model.predict(X_train)
    train_accuracy = metrics.accuracy_score(Y_train, train_pred)
    print("train_accuracy: ", train_accuracy)

    # Test accuracy
    test_pred = model.predict(X_test)
    test_accuracy = metrics.accuracy_score(Y_test, test_pred)
    print("test_accuracy: ", test_accuracy)

    model.fit(encoded_train1, labels)

    predictions = model.predict_proba(encoded_test1)[:, 1]
    test_predictions = pd.DataFrame({'ID': id_list, 'Prediction': predictions})
    test_predictions.to_csv("./results/cat_boost_prob.csv", index=False)

    predictions = model.predict(encoded_test1)
    test_predictions = pd.DataFrame({'ID': id_list, 'Prediction': predictions})
    test_predictions.to_csv("./results/cat_boost.csv", index=False)


get_catboost_prediction()
