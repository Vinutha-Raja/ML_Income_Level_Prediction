import numpy as np
from sklearn import tree
from sklearn import preprocessing
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.datasets import make_classification
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer


attr_numerical = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week",]
attr_list = ["age", "workclass", "fnlwgt", "education", "education-num",
                                                         "marital-status", "occupation", "relationship",
                                                         "race", "sex", "capital-gain", "capital-loss",
                                                         "hours-per-week", "native-country", "label"]
attr_index_map = {}
index = 0
for att in attr_list:
    attr_index_map[att] = index
    index += 1

test_attr_list = ["age", "workclass", "fnlwgt", "education", "education-num",
                                                         "marital-status", "occupation", "relationship",
                                                         "race", "sex", "capital-gain", "capital-loss",
                                                         "hours-per-week", "native-country"]
test_attr_map = {}
index = 0
for att in test_attr_list:
    test_attr_map[att] = index
    index += 1


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


def adaboost_algo(train, test, labels):
    clf = AdaBoostClassifier(n_estimators=400, random_state=1, learning_rate=1.0)
    clf.fit(train, labels)
    predictions = clf.predict(test)
    return predictions


def encode_categorical_data(data, att_list):
    categorical_attr = ["workclass", "education", "marital-status", "occupation", "relationship",
                                                             "race", "sex", "native-country"]
    for attr in att_list:
        le = preprocessing.LabelEncoder()
        data[attr] = le.fit_transform(data[attr].astype(str))
        # le.fit(data[attr].tolist())
        # data[attr] = le.transform(data[attr].tolist())
    # print(data)
    return data


def handle_missing_data_manually(df, attr_map, att_list):
    # print(attr_map, att_list)
    for attr in att_list:
        # if '?' in df[attr].unique():

        for i in range(len(df.index)):
            # print(df.iloc[i, attr_map[attr]])
            if df.iloc[i, attr_map[attr]] == "?":
                # print("yes")
                label = df.iloc[i, attr_map['label']]
                new_df = df[df['label'] == label]
                majority_list = new_df[attr].value_counts().index.tolist()[:1]
                majority = majority_list[0]
                # print("majority", majority)
                df.iloc[i, attr_map[attr]] = majority
    return df


def handle_missing_data(data):
    imp = SimpleImputer(strategy="most_frequent")
    data = imp.fit_transform(data)
    return pd.DataFrame(data, columns=attr_list)


def handle_test_missing_data(data):
    imp = SimpleImputer(strategy="most_frequent")
    data = imp.fit_transform(data)
    return pd.DataFrame(data, columns=test_attr_list)


def fill_missing_values_with_majority(data):
    data['workclass'] = data['workclass'].fillna(data['workclass'].mode()[0])
    data['occupation'] = data['occupation'].fillna(data['occupation'].mode()[0])
    data['native-country'] = data['native-country'].fillna(data['native-country'].mode()[0])
    return data


def get_decision_tree_predictions():
    train = read_df_from_csv("./income2022f/train_final.csv")
    train = train.replace('?', np.nan)
    train = fill_missing_values_with_majority(train)

    train = train.drop_duplicates(keep='first')
    train = encode_categorical_data(train, attr_list)
    # train = convert_numerical_columns(train)

    # print(train)
    # train = handle_missing_data(train)
    # train = handle_missing_data_manually(train, attr_index_map, attr_list)
    # print(train)

    # train = handle_missing_data(train)

    labels = train['label'].tolist()
    train = train.drop(columns=['label'])

    test = read_df_from_csv("./income2022f/test_final.csv", True)
    print("test shape", test.shape)
    test = test.drop(columns=['ID'])
    # test = convert_numerical_columns(test)
    # test = test.replace('?', np.nan)
    # print("nan", test.isna().sum())
    print(test)
    # test = fill_missing_values_with_majority(test)
    # test = handle_test_missing_data(test)
    # test = handle_missing_data_manually(test, test_attr_map, test_attr_list)
    print(test)
    # Encoding categorical data

    test = encode_categorical_data(test, test_attr_list)
    print(test)
    train_list = train.values.tolist()
    test_list = test.values.tolist()

    print(len(test_list))
    # print(train_list, test_list)
    predictions = adaboost_algo(train_list, test_list, labels)
    print(len(predictions))
    result_list = []
    i = 1
    for pred in predictions:
        result_list.append([i, pred])
        i += 1
    print(len(result_list))
    result_df = pd.DataFrame(data=result_list)
    print(result_df.shape)
    result_df.to_csv("./adaboost_result_mode_400.csv", index=False, header=["ID", "Prediction"])
    print(predictions)


get_decision_tree_predictions()