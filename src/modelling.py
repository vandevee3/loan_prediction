from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from utils import serialize_data, deserialize_data, train_model, thresholding_tuning, choose_threshold, model_evaluation

import yaml

def modelling():
    with open ('../config/path.yaml', 'r') as f:
        path = yaml.safe_load(f)

    X_TRAIN_ROS_PATH = path['data_path']['data']['processed']['X_train_ros']
    Y_TRAIN_ROS_PATH = path['data_path']['data']['processed']['y_train_ros']
    X_VALID_PATH = path['data_path']['data']['processed']['X_valid_prep']
    Y_VALID_PATH = path['data_path']['output']['y_valid']
    X_TEST_PATH = path['data_path']['data']['processed']['X_test_prep']
    Y_TEST_PATH = path['data_path']['output']['y_test']

    LOGREG_MODEL_PATH = path['data_path']['models']['logreg_model']
    DESTREE_MODEL_PATH = path['data_path']['models']['destree_model']
    RANDFOR_MODEL_PATH = path['data_path']['models']['randfor_model']
    SUPVEC_MODEL_PATH = path['data_path']['models']['supvec_model']

    LOGREG_THRESHOLD_PATH = path['param_path']['threshold']['logistic_regression']
    DESTREE_THRESHOLD_PATH = path['param_path']['threshold']['decision_tree']
    RANDFOR_THRESHOLD_PATH = path['param_path']['threshold']['random_forest']
    SUPVEC_THRESHOLD_PATH = path['param_path']['threshold']['super_vector']

    X_train_ros = deserialize_data(X_TRAIN_ROS_PATH)
    y_train_ros = deserialize_data(Y_TRAIN_ROS_PATH)
    X_valid_prep = deserialize_data(X_VALID_PATH)
    y_valid = deserialize_data(Y_VALID_PATH)
    X_test_prep = deserialize_data(X_TEST_PATH)
    y_test = deserialize_data(Y_TEST_PATH)

    logreg_hyperparameter = {
    'solver'  : ['saga', 'liblinear'],
    'penalty' : ['l2', 'l1'],
    'max_iter' : [500, 1000, 2000]
    }

    decistree_hyperparameter = {
        'max_depth' : [5, 15],
        'min_samples_split' : [2, 10],
        'min_samples_leaf' : [1, 5],
        'max_features' : [None, 'sqrt'],
    }

    randforest_hyperparameter ={
        'n_estimators' : [100, 200],
        'max_depth' : [None, 10],
        'min_samples_split' : [2, 10],
        'max_leaf_nodes' : [None, 3],
        'n_estimators': [100, 200]
    }

    supvec_hyperparameter = {
        'C' : [0.1, 10],
        'kernel' : ['rbf'],
    }

    dummy_clf = DummyClassifier(strategy= "stratified", random_state= 42)
    base_model = train_model(dummy_clf, {}, X_train_ros, y_train_ros)

    logreg_instance = LogisticRegression(
    random_state = 42
    )

    destree_instance  = DecisionTreeClassifier(
        random_state = 42
    )

    randfor_instance  = RandomForestClassifier(
        random_state = 42
    )

    supvec_instance = SVC(
        random_state = 42,
        probability = True
    )

    logreg_model = train_model(logreg_instance, logreg_hyperparameter, X_train_ros, y_train_ros)
    destree_model = train_model(destree_instance, decistree_hyperparameter, X_train_ros, y_train_ros)
    randfor_model = train_model(randfor_instance, randforest_hyperparameter, X_train_ros, y_train_ros)
    # supvec_model = train_model(supvec_instance, supvec_hyperparameter, X_train_ros, y_train_ros)

    serialize_data(logreg_model, LOGREG_MODEL_PATH)
    serialize_data(destree_model, DESTREE_MODEL_PATH)
    serialize_data(randfor_model, RANDFOR_MODEL_PATH)
    # serialize_data(supvec_model, SUPVEC_MODEL_PATH)

    thresh_baseline = thresholding_tuning(base_model, X_valid_prep, y_valid)
    thresh_logreg = thresholding_tuning(logreg_model, X_valid_prep, y_valid)
    thresh_destree = thresholding_tuning(destree_model, X_valid_prep, y_valid)
    thresh_randfor = thresholding_tuning(randfor_model, X_valid_prep, y_valid)
    # thresh_supvec = thresholding_tuning(supvec_model, X_valid_prep, y_valid)

    best_thresh_logreg = choose_threshold(thresh_logreg, LOGREG_THRESHOLD_PATH)
    best_thresh_destree = choose_threshold(thresh_destree, DESTREE_THRESHOLD_PATH)
    best_thresh_randfor = choose_threshold(thresh_randfor, RANDFOR_THRESHOLD_PATH)
    # best_thresh_supvec = choose_threshold(thresh_supvec, SUPVEC_THRESHOLD_PATH)

    logreg_eval = model_evaluation(logreg_model, best_thresh_logreg['threshold'], X_test_prep, y_test)
    destree_eval = model_evaluation(destree_model, best_thresh_destree['threshold'], X_test_prep, y_test)
    randfor_eval = model_evaluation(randfor_model, best_thresh_randfor['threshold'], X_test_prep, y_test)
    # supvec_eval = model_evaluation(supvec_model, best_thresh_supvec['threshold'], X_test_prep, y_test)

    # print(f"Logistic Regression Model Evaluation : \n {logreg_eval}")
    # print(f"Decision Tree Model Evaluation : \n {destree_eval}")
    # print(f"Random Forest Model Evaluation : \n {randfor_eval}")
    # print(f"Support Vector Classification Model Evaluation : \n {supvec_eval}")


if __name__ == "__main__":
    modelling()