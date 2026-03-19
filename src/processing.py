from .utils import deserialize_data, create_onehot_encoder, ohe_transform
import yaml
from pathlib import Path

def ohe_transform_full(df):
    BASE_DIR = Path(__file__).resolve().parent.parent  
    # CONFIG_PATH = BASE_DIR / "config" / "path.yaml"

    # with open(CONFIG_PATH, 'r') as f:
    #     path = yaml.safe_load(f)

    OHE_HOME_OWNERSHIP_PATH  = BASE_DIR / "models" / "ohe_home_ownership.pkl"
    OHE_DEFAULT_ON_FILE_PATH = BASE_DIR / "models" / "ohe_default_on_file.pkl"
    OHE_LOAN_GRADE_PATH      = BASE_DIR / "models" / "ohe_loan_grade.pkl"
    OHE_LOAN_INTENT_PATH     = BASE_DIR / "models" / "ohe_loan_intent.pkl"

    ohe_transform_ownership = deserialize_data(OHE_HOME_OWNERSHIP_PATH)
    ohe_loan_intent = deserialize_data(OHE_LOAN_INTENT_PATH)
    ohe_loan_grade = deserialize_data(OHE_LOAN_GRADE_PATH)
    ohe_default_on_file = deserialize_data(OHE_DEFAULT_ON_FILE_PATH)

    df = ohe_transform(df, 'person_home_ownership', 'home_ownership', ohe_transform_ownership)
    df = ohe_transform(df, 'loan_intent', 'loan_intent', ohe_loan_intent)
    df = ohe_transform(df, 'loan_grade', 'loan_grade', ohe_loan_grade)
    df = ohe_transform(df, 'cb_person_default_on_file', 'default_onfile', ohe_default_on_file)

    return df