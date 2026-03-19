from .utils import serialize_data, deserialize_data, drop_duplicate_data, median_imputation, create_onehot_encoder, ohe_transform
import yaml

def data_preprocess():
    with open('../config/path.yaml', 'r') as f:
        path = yaml.safe_load(f)

    X_train_path = path['data_path']['input']['X_train']
    x_test_path = path['data_path']['input']['x_test']
    x_valid_path = path['data_path']['input']['x_valid']
    y_train_path = path['data_path']['output']['y_train']
    X_train_prep_path = path['data_path']['data']['processed']['X_train_prep']
    X_test_prep_path = path['data_path']['data']['processed']['X_test_prep']
    X_valid_prep_path = path['data_path']['data']['processed']['X_valid_prep']
    y_train_prep_path = path['data_path']['data']['processed']['y_train_prep']

    ohe_home_ownership_path = path['data_path']['models']['ohe_home_ownership']
    ohe_loan_intent_path = path['data_path']['models']['ohe_loan_intent']
    ohe_loan_grade_path = path['data_path']['models']['ohe_loan_grade']
    ohe_default_on_file_path = path['data_path']['models']['ohe_default_on_file']

    X_TRAIN_PATH = X_train_path
    X_TEST_PATH = x_test_path
    X_VALID_PATH = x_valid_path
    Y_TRAIN_PATH = y_train_path
    X_TRAIN_PREP_PATH = X_train_prep_path
    X_TEST_PREP_PATH = X_test_prep_path
    X_VALID_PREP_PATH = X_valid_prep_path
    Y_TRAIN_PREP_PATH = y_train_prep_path

    OHE_HOME_OWNERSHIP_PATH = ohe_home_ownership_path
    OHE_LOAN_INTENT_PATH = ohe_loan_intent_path
    OHE_LOAN_GRADE_PATH = ohe_loan_grade_path
    OHE_DEFAULT_FILE_ON_PATH = ohe_default_on_file_path

    X_train = deserialize_data(X_TRAIN_PATH)
    X_test = deserialize_data(X_TEST_PATH)
    X_valid = deserialize_data(X_VALID_PATH)
    y_train = deserialize_data(Y_TRAIN_PATH)

    X_train, y_train = drop_duplicate_data(X_train, y_train)
    subset_data = X_train.columns[X_train.isna().any()].to_list()
    subset_data = median_imputation(X_train, subset_data, True)
    X_train = median_imputation(X_train, subset_data, False)
    X_test = median_imputation(X_test, subset_data, False)
    X_valid = median_imputation(X_valid, subset_data, False)

    person_home_ownership = X_train['person_home_ownership'].to_list()
    loan_intent = X_train['loan_intent'].to_list()
    loan_grade = X_train['loan_grade'].to_list()
    cb_person_default_on_file = X_train['cb_person_default_on_file'].to_list()

    ohe_home_ownership = create_onehot_encoder(person_home_ownership, OHE_HOME_OWNERSHIP_PATH)
    ohe_loan_intent = create_onehot_encoder(loan_intent, OHE_LOAN_INTENT_PATH)
    ohe_loan_grade = create_onehot_encoder(loan_grade, OHE_LOAN_GRADE_PATH)
    ohe_default_on_file = create_onehot_encoder(cb_person_default_on_file, OHE_DEFAULT_FILE_ON_PATH)

    X_train = ohe_transform(X_train, 'person_home_ownership', 'home_ownership', ohe_home_ownership)
    X_train = ohe_transform(X_train, 'loan_intent', 'loan_intent', ohe_loan_intent)
    X_train = ohe_transform(X_train, 'loan_grade', 'loan_grade', ohe_loan_grade)
    X_train = ohe_transform(X_train, 'cb_person_default_on_file', 'default_onfile', ohe_default_on_file)
    X_test = ohe_transform(X_test, 'person_home_ownership', 'home_ownership', ohe_home_ownership)
    X_test = ohe_transform(X_test, 'loan_intent', 'loan_intent', ohe_loan_intent)
    X_test = ohe_transform(X_test, 'loan_grade', 'loan_grade', ohe_loan_grade)
    X_test = ohe_transform(X_test, 'cb_person_default_on_file', 'default_onfile', ohe_default_on_file)
    X_valid = ohe_transform(X_valid, 'person_home_ownership', 'home_ownership', ohe_home_ownership)
    X_valid = ohe_transform(X_valid, 'loan_intent', 'loan_intent', ohe_loan_intent)
    X_valid = ohe_transform(X_valid, 'loan_grade', 'loan_grade', ohe_loan_grade)
    X_valid = ohe_transform(X_valid, 'cb_person_default_on_file', 'default_onfile', ohe_default_on_file)
    serialize_data(X_train, X_TRAIN_PREP_PATH)
    serialize_data(X_test, X_TEST_PREP_PATH)
    serialize_data(X_valid, X_VALID_PREP_PATH)
    serialize_data(y_train, Y_TRAIN_PREP_PATH)


if __name__ == "__main__":
    data_preprocess()



