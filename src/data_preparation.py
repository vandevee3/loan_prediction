from .utils import load_data, split_input_output, split_train_test, serialize_data, deserialize_data
import yaml

def data_prep():
    with open('../config/path.yaml', 'r') as f:
        path = yaml.safe_load(f)

    X_train_path = path['data_path']['input']['X_train']
    x_test_path = path['data_path']['input']['x_test']
    x_valid_path = path['data_path']['input']['x_valid']
    y_train_path = path['data_path']['output']['y_train']
    y_test_path = path['data_path']['output']['y_test']
    y_valid_path = path['data_path']['output']['y_valid']
    fname = path['data_path']['data']['raw']['credit_risk']

    FNAME = fname
    TARGET_COL = "loan_status"
    X_TRAIN_PATH = X_train_path
    X_TEST_PATH = x_test_path
    X_VALID_PATH = x_valid_path
    Y_TRAIN_PATH = y_train_path
    Y_TEST_PATH = y_test_path
    Y_VALID_PATH = y_valid_path


    data = load_data(FNAME)
    X,y = split_input_output(data, target_col= TARGET_COL)
    X_train, X_non_train, y_train, y_non_train = split_train_test(X,y, test_size= 0.2, random_state= 42)
    X_valid, X_test, y_valid, y_test = split_train_test(X_non_train, y_non_train, test_size= 0.5, random_state= 42)
    serialize_data(X_train, X_TRAIN_PATH)
    serialize_data(X_test, X_TEST_PATH)
    serialize_data(X_valid, X_VALID_PATH)
    serialize_data(y_train, Y_TRAIN_PATH)
    serialize_data(y_test, Y_TEST_PATH)
    serialize_data(y_valid, Y_VALID_PATH)

if __name__ == "__main__":
    data_prep()
    