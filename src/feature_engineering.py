from utils import serialize_data, deserialize_data, random_oversampling
import yaml 

def feature_engineering():
    with open('../config/path.yaml', 'r') as f:
        path = yaml.safe_load(f)

    X_train_prep_path = path['data_path']['data']['processed']['X_train_prep']
    y_train_prep_path = path['data_path']['data']['processed']['y_train_prep']
    X_train_ros_path = path['data_path']['data']['processed']['X_train_ros']
    y_train_ros_path = path['data_path']['data']['processed']['y_train_ros']

    X_TRAIN_PREP_PATH = X_train_prep_path
    Y_TRAIN_PREP_PATH = y_train_prep_path
    X_TRAIN_ROS_PATH = X_train_ros_path
    Y_TRAIN_ROS_PATH = y_train_ros_path

    X_train_prep = deserialize_data(X_TRAIN_PREP_PATH)
    y_train_prep = deserialize_data(Y_TRAIN_PREP_PATH)

    X_train_ros, y_train_ros = random_oversampling(X_train_prep, y_train_prep)

    serialize_data(X_train_ros, X_TRAIN_ROS_PATH)
    serialize_data(y_train_ros, Y_TRAIN_ROS_PATH)

if __name__ == '__main__':
    feature_engineering()