from .utils import deserialize_data, create_onehot_encoder, ohe_transform
import yaml

def ohe_transform_full(df):
    with open('../config/path.yaml', 'r') as f:
        path = yaml.safe_load(f)

    ohe_home_ownership_path = path['data_path']['models']['ohe_home_ownership']
    ohe_loan_intent_path = path['data_path']['models']['ohe_loan_intent']
    ohe_loan_grade_path = path['data_path']['models']['ohe_loan_grade']
    ohe_default_on_file_path = path['data_path']['models']['ohe_default_on_file']

    OHE_HOME_OWNERSHIP_PATH = ohe_home_ownership_path
    OHE_LOAN_INTENT_PATH = ohe_loan_intent_path
    OHE_LOAN_GRADE_PATH = ohe_loan_grade_path
    OHE_DEFAULT_FILE_ON_PATH = ohe_default_on_file_path

    ohe_transform_ownership = deserialize_data(OHE_HOME_OWNERSHIP_PATH)
    ohe_loan_intent = deserialize_data(OHE_LOAN_INTENT_PATH)
    ohe_loan_grade = deserialize_data(OHE_LOAN_GRADE_PATH)
    ohe_default_on_file = deserialize_data(OHE_DEFAULT_FILE_ON_PATH)

    df = ohe_transform(df, 'person_home_ownership', 'home_ownership', ohe_transform_ownership)
    df = ohe_transform(df, 'loan_intent', 'loan_intent', ohe_loan_intent)
    df = ohe_transform(df, 'loan_grade', 'loan_grade', ohe_loan_grade)
    df = ohe_transform(df, 'cb_person_default_on_file', 'default_onfile', ohe_default_on_file)

    return df