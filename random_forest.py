from sklearn import preprocessing
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

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


def random_forest_algo(train, test, labels):
    clf = RandomForestClassifier(random_state=1, n_estimators=1000, criterion="log_loss", max_depth=30, max_features=10)
    # max_depth = None,max_feature
    clf.fit(train, labels)
    predictions = clf.predict(test)
    return predictions


def encode_categorical_data(data):
    categorical_attr = ["workclass", "education", "marital-status", "occupation", "relationship",
                                                             "race", "sex", "native-country"]
    for attr in categorical_attr:
        le = preprocessing.LabelEncoder()
        le.fit(data[attr].tolist())
        data[attr] = le.transform(data[attr].tolist())
    # print(data)
    return data


def get_decision_tree_predictions():
    train = read_df_from_csv("./income2022f/train_final.csv")
    train = convert_numerical_columns(train)
    labels = train['label'].tolist()
    train = train.drop(columns=['label'])
    test = read_df_from_csv("./income2022f/test_final.csv", True)
    print("test shape", test.shape)
    test = test.drop(columns=['ID'])

    test = convert_numerical_columns(test)

    # Encoding categorical data
    train = encode_categorical_data(train)
    test = encode_categorical_data(test)

    train_list = train.values.tolist()
    test_list = test.values.tolist()
    print(len(test_list))
    # print(train_list, test_list)
    predictions = random_forest_algo(train_list, test_list, labels)
    print(len(predictions))
    result_list = []
    i = 1
    for pred in predictions:
        result_list.append([i, pred])
        i += 1
    print(len(result_list))
    result_df = pd.DataFrame(data=result_list)
    print(result_df.shape)
    result_df.to_csv("./random_forest_log_loss_1k.csv", index=False, header=["ID", "Prediction"])
    print(predictions)


get_decision_tree_predictions()