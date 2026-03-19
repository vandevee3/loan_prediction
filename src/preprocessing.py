import pandas as pd
from sklearn.preprocessing import OneHotEncoder

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