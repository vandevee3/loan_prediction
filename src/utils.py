import pandas as pd
import joblib
import numpy as np
import json

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from copy import deepcopy
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, FixedThresholdClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, classification_report, make_scorer, roc_auc_score, RocCurveDisplay



def load_data(fname: str):
    """
    To load data from desired path

    Paramaters :
        - fname : File path

    Return value:
        - data : Data wrapped in pandas dataframe
    """

    try:
        with open(fname, 'r'):
            pass
        print("File exist and is readable")
    
    except FileNotFoundError:
        print("File doesn't exist")
    
    except PermissionError:
        print("File exists but no permission to access")

    if fname.endswith('.csv'):
        data = pd.read_csv(fname)
        print(f"Data Shape : {data.shape}")

    elif fname.endswith('.xlsx'):
        data = pd.read_excel(fname)
        print(f"Data Shape : {data.shape}")
    
    else :
        data = pd.DataFrame()
        print("Can't read from others type of file, data will set into empty dataframe")
    
    return data

def split_input_output(data: pd.DataFrame, target_col: str):
    """ 
    Splitting input and output from data 

    Parameters:
        - data: Pandas dataframe
        - target_col: string

    Return Value:
        - X : Input in pandas dataframe
        - y : Output in pandas dataframe
    """

    X = data.drop(columns= target_col)
    y = data[target_col]

    print(f"Original data shape: {data.shape}")
    print(f"X data shape: {X.shape}")
    print(f"y data shape: {y.shape}")

    return X, y

def split_train_test(X: pd.DataFrame, y: pd.DataFrame, test_size: float, random_state: int):
    """
    Splitting between train and test data

    Parameters:
        - X : Data as an input in pandas dataframe
        - y : Data as an output in pandas dataframe
        - test_size : Propotion of test data, if you set 0.2 then train data will have 0.8 of the data and the type is float
        - random_state : Random number generator to shuffle the data and the type is integer

    Return Value: 
        - X_train : Data for input train in pandas dataframe
        - X_test : Data for input test in pandas dataframe
        - y_train : Data for output train in pandas dataframe
        - y_test : Data for output test in pandas dataframe
    """

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= test_size, random_state= random_state, stratify= y)
    print(f"X train shape: {X_train.shape}")
    print(f"X test shape: {X_test.shape}")
    print(f"y train shape: {y_train.shape}")
    print(f"y test shape: {y_test.shape}")

    return X_train, X_test, y_train, y_test

def serialize_data(data: pd.DataFrame, path: str):
    """
    Serialize data into pickle file

    Parameters:
        - data : Data in pandas dataframe format
        - path : Path used to save serialize data in string

    Return Value:
        None, serialize data will automatically saved into configured path
    """

    try: 
        joblib.dump(data, path)
        print("Succeed to serialize data")
    except:
        print("Failed to serialize data!")

def deserialize_data(path: str):
    """
    Desrialize data from existing file

    Parameters:
        - path : Path used to load serializedata in string

    Return Value:
        - data : contains deserialize data in pandas dataframe
    """

    try:
        print(f"Loading model from: {path}") 
        data = joblib.load(path)
        return data
    except Exception as e:
        print(f"Failed to deserialize data! Error: {e}")
        raise

def drop_duplicate_data(X: pd.DataFrame, y: pd.Series):
    """ 
    To drop duplicated datas

    Parameters:
        X : Input data frame (type: pandas DataFrame)
        y : Output / target data frame (type: pandas DataFrame)

    Return Value:
        X : Input data frame with no duplicated datas (type: pandas DataFrame)
        y : Output / target data frame with no duplicated datas (type: pandas DataFrame)
    """
    
    if type(X) != pd.DataFrame:
        raise RuntimeError("X type not valid!")
    elif type(y) != pd.Series:
        raise RuntimeError("y type not valid!")
    else :
        print("Fungsi drop_duplicate_data: parameter telah divalidasi")

    X = X.copy()
    y = y.copy()
    print(f"Fungsi drop_duplicate_data: shape dataset sebelum dropping duplicate X adalah : {X.shape}")
    X_duplicate = X[X.duplicated(keep= False)]
    print(f"Fungsi drop_duplicate_data: shape dari data X yang duplicate adalah: {X_duplicate.shape}")
    X_clean = tuple()
    X_clean = len(X) - len(X_duplicate)
    print(f"Fungsi drop_duplicate_data: shape dataset X setelah drop duplicate seharusnya adalah: {X_clean}")
    X.drop_duplicates(inplace= True, keep= False)
    print(f"Fungsi drop_duplicate_data: shape dataset X setelah dropping duplicate adalah: {X.shape}")
    y = y.loc[X.index]
    print(f"Fungsi drop_duplicate_data: shape dataset y setelah dropping duplicate adalah: {y.shape}")
    
    return X, y

def median_imputation(data: pd.DataFrame, subset_data, fit: bool):
    """ 
    To calculate or do imputation with median

    Parameters:
        - data : input data (type: pandas DataFrame)
        - subset_data : it depends on fit, if fit = True then it contains columns name with nan value, and if fit = False then it will impute based on subset_data
            (
                type :
                    if fit = True, list
                    if fit = False, dict
            )

    Return Value:
        - if fit = True:
            imputation_data : contains median for each selected column (type: float64)
        - if fit = False:
            data : data with imputed nan column (type: pandas DataFrame)
    """

    if not isinstance(data, pd.DataFrame):
        raise RuntimeError('Fungsi median_imputation: parameter data haruslah bertipe DataFrame!')
    
    if fit == True:
        if not isinstance(subset_data, list):
            raise RuntimeError('Fungsi median_imputation: untuk nilai parameter fit = True, ' \
            'subset_data harus bertipe list dan berisi daftar nama kolom yang ingin dicari nilai mediannya guna menjadi data imputasi pada kolom tersebut.')
        
        print('')
        print("Fungsi median_imputation: parameter telah divalidasi.")
        data = data.copy()
        subset_data = deepcopy(subset_data) # sepertinya menggunakan copy(deep= True) juga bisa bekerja
        imputation_data = dict()

        for subset in subset_data:
            med = data[subset].median()
            imputation_data[subset] = med

        print(f"Fungsi median_imputation: proses fitting telah selesai, berikut hasilnya {imputation_data}")

        return imputation_data
            

    elif fit == False:
        if not isinstance(subset_data, dict):
            raise RuntimeError('Fungsi median_imputation: untuk nilai parameter fit = False, ' \
            'subset_data harus bertipe dict dan berisi key yang merupakan nama kolom beserta value yang merupakan nilai median dari kolom tersebut.')
        
        print('')
        print("Fungsi median_imputation: parameter telah divalidasi.")
        data = data.copy()
        subset_data = deepcopy(subset_data) # sepertinya menggunakan copy(deep= True) juga bisa bekerja

        for subset in subset_data:
            print('Fungsi median_imputation: informasi count na sebelum dilakukan imputasi:')
            print(f"{data[subset].isna().sum()}")
            print('')

        data.fillna(subset_data, inplace= True)
        
        for subset in subset_data:
            print('Fungsi median_imputation: informasi count na setelah dilakukan imputasi:')
            print(f"{data[subset].isna().sum()}")
            print('')

        return data

    else:
        raise RuntimeError('Fungsi median_imputation: parameter fit haruslah bertipe boolean, bernilai True atau False.')
    
def create_onehot_encoder(categories: list, path: str):
    """ 
    Onehot Encoder for categorical data

    Parameters:
        - categories : categorical data (type: list)
        - path : path to saved ohe (type: str)

    Return Value:
        - ohe : onehot encoder (type: sklearn OneHotEncoder)
    """
    
    if not isinstance(categories, list):
        raise RuntimeError('Fungsi create_onehot_encoder: parameter categories haruslah bertipe list, berisi kategori yang akan dibuat encodernya.')
    elif any(isinstance(item, list) for item in categories):
        raise ValueError('Fungsi create_onehot_encoder: parameter categories memiliki dimensi lebih dari 1!')

    if not isinstance(path, str):
        raise RuntimeError('Fungsi create_onehot_encoder: parameter path haruslah bertipe string, berisi lokasi pada disk komputer dimana encoder akan disimpan.')
    
    ohe = OneHotEncoder()
    categories = np.array(categories).reshape(-1, 1)
    ohe.fit(categories)
    serialize_data(ohe, path)
    print(f"Kategori yang telah dipelajari adalah {ohe.categories_[0].tolist()}")

    return ohe

def ohe_transform(dataset: pd.DataFrame, subset: str, prefix: str, ohe: OneHotEncoder):
    """  
    Transforing OHE into dataset

    Parameters :
        - dataset : input data (type: pandas Dataframe)
        - subset : selected column to implement transformation (type: string)
        - prefix : prefix for OHE transformation (type: string)
        - ohe : selected OneHotEncoder (type: sklearn OneHotEncoder)

    Return Value:
        - dataset : transformed data with OHE (type: pandas DataFrame)
    """

    if not isinstance(dataset, pd.DataFrame):
        raise RuntimeError('Fungsi ohe_transform: parameter dataset harus bertipe DataFrame!')
    
    if not isinstance(ohe, OneHotEncoder):
        raise RuntimeError('Fungsi ohe_transform: parameter ohe harus bertipe OneHotEncoder!')
    
    if not isinstance(prefix, str):
        raise RuntimeError('Fungsi ohe_transform: parameter prefix harus bertipe str!')
    
    if not isinstance(subset, str):
        raise RuntimeError('Fungsi ohe_transform: parameter subset harus bertipe str!')
    
    try:
        column_list = dataset.columns.to_list()
        column_list.index(subset)
    except:
        raise RuntimeError('Fungsi ohe_transform: parameter subset string namun data tidak ditemukan dalam daftar kolom yang terdapat pada parameter dataset.')
    
    print('Fungsi ohe_transform: parameter telah divalidasi.')
    dataset = dataset.copy()
    print(f"Fungsi ohe_transform: daftar nama kolom sebelum dilakukan pengkodean adalah {dataset.columns.to_list()}")
    col_names = [prefix + '_' + col_name for col_name in ohe.categories_[0].tolist()]
    
    encoded = pd.DataFrame(ohe.transform(dataset[[subset]]).toarray(), columns= col_names, index= dataset.index)
    dataset = pd.concat([dataset, encoded], axis= 1)
    dataset.drop(columns= [subset], inplace= True)
    print(f"Fungsi ohe_transform: daftar nama kolom setelah dilakukan pengkodean adalah {dataset.columns}")
    return dataset

def random_oversampling(X_train: pd.DataFrame, y_train: pd.DataFrame):
    """
    Random over sampling to hanlde imbalance data

    Parameters:
        - X_train : input training data (type: pandas Dataframe)
        - y_train : output training data (type: pandas Dataframe)

    Return Values:
        - X_train_ros : input training data with random over sampling (type: pandas Dataframe)
        - y_train_ros : output training data with random over sampling (type: pandas Dataframe)
    """
    ros = RandomOverSampler()
    X_train_ros, y_train_ros = ros.fit_resample(X_train, y_train)

    return X_train_ros, y_train_ros

def train_model(model, hyperparam: dict, input_train: pd.DataFrame, output_target: pd.DataFrame):

    """
    To train model 

    Parameters:
        - model : an instance of model (type: instance)
        - hyperparameter : hyperparameter for model cross-validation (type: dict)
        - input_train : input data for training (type: pandas dataftame)
        - output_target : target data for training (type: pandas dataframe)

    Return Values:
        - grid_search : model that fit using Grid Search CV (type: class) 
    """

    compute_support_models = ['LogisticRegression', 'DecisionTreeClassifier', 'SVC']
    scoring = {
        'recall' : 'recall',
        'precision' : 'precision',
        'f1' : 'f1'
    }

    print(f"Model name : {model.__class__.__name__}")
    if model.__class__.__name__ in compute_support_models:
        grid_search = GridSearchCV(
            estimator = model,
            param_grid = hyperparam,
            n_jobs = -1,
            verbose = 3,
            cv = 5,
            scoring = scoring,
            refit = 'recall'
        )
    else :
        grid_search = GridSearchCV(
            estimator = model,
            param_grid = hyperparam,
            verbose = 3,
            cv = 5,
            scoring = scoring,
            refit = 'recall'
        )

    grid_search.fit(input_train, output_target)

    return grid_search

def thresholding_tuning(model, input_train : pd.DataFrame, output_train : pd.DataFrame):

    """ 
    Tuning threshold for model

    Parameters:
        - model : model of machine learning (type: instance)
        - input_train : input data for thresholding (type: pandas dataframe)
        - output_train : output data for thresholding (type: pandas dataframe)

    Return Values:
        - threshold_result : threshold and metric value used (type: list)
    """

    list_threshold = np.linspace(0, 1, 100)
    threshold_result = []
    print(f"Model Name : {model.best_estimator_.__class__.__name__}")

    for threshold in list_threshold:

        tunned_threshold = FixedThresholdClassifier(
            estimator = model,
            threshold = threshold
        )

        y_pred_tunned = tunned_threshold.predict(input_train)
        recall_tunned = recall_score(output_train, y_pred_tunned)
        print(f"Threshold : {round(threshold, 4)} | Recall : {round(recall_tunned, 4)}")
        threshold_result.append({'threshold' : round(threshold, 4), 'recall' : round(recall_tunned, 4)})

    # plt.plot(list_threshold, [i['recall'] for i in threshold_result])
    # plt.show()

    return threshold_result

def choose_threshold(thershold_result: list, path: str):
    """
    To choose best threshold 

    Parameters:
        - threshold_result : contains threshold and metric value (type: list)
        - path : path used to save best threshold (type: string)

    Return Values:
        - best_thresh : contains best threshold and metric value (type: list)
    """
    
    thershold_result = [thershold_result]
    thresh_list = []
    min_recall = 0.8

    for thresh_element in thershold_result:
        thresh_list.extend(thresh_element)

    df = pd.DataFrame(thresh_list)      
    df_filtered = df[df["recall"] >= min_recall]                      
    df_sorted   = df_filtered.sort_values("threshold", ascending=False)
    best_thresh  = df_sorted.iloc[0].to_dict()

    print("BEST THRESHOLD: ")
    print(f"  Threshold   : {best_thresh['threshold']}")
    print(f"  Recall      : {best_thresh['recall']:.4f}")

    best_threshold_serialize = {
        "threshold" : best_thresh['threshold'],
        "recall" : best_thresh['recall']
    }

    with open(path, 'w') as f:
        json.dump(best_threshold_serialize, f, indent= 4)       
          
    return best_thresh

def model_evaluation(model, threshold : list, input_train : pd.DataFrame, output_train : pd.DataFrame):

    """ 
    Evaluate model 

    Parameters:
        - model : model of machine learning (type: instance)
        - threshold : best threshold from tuning threshold (type: list)
        - input_train : input data for thresholding (type: pandas dataframe)
        - output_train : output data for thresholding (type: pandas dataframe)

    Return Values:
        - report_dict : report of model evaluation (type: dict)
    """

    y_prob = model.predict_proba(input_train)
    y_pred = []
    for prob in y_prob:
        best_class = np.argmax(prob)
        best_prob  = prob[best_class]

        if best_prob >= threshold:
            y_pred.append(best_class)   
        else:
            y_pred.append(1)   

    y_pred = np.array(y_pred)

    report_str = classification_report(
        output_train, y_pred,
        zero_division=0
    )
    print(f"Model Name : {model.best_estimator_.__class__.__name__}")
    print(report_str)

    report_dict = classification_report(
        output_train, y_pred,
        output_dict=True,
        zero_division=0
    )
    # auc = roc_auc_score(output_train, y_prob[:,1])
    # print(f"Model : {auc}")
    # RocCurveDisplay.from_predictions(output_train, y_prob[:, 1])
    # plt.show()

    return report_dict