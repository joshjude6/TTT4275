import pandas as pd

def get_data(filename='', header=0) -> pd.DataFrame:
    return pd.read_csv(filename, header=header)

def separate_data(data, training_samples=-1) -> tuple[list, list]:
    training_data = data[0:training_samples]
    test_data = data[training_samples:]
    return training_data, test_data

def drop_entry_columns(data, column_names) -> pd.DataFrame:
    data.columns = data.columns.str.strip()
    data = data.drop(columns=column_names)
    return data

if __name__ == '__main__':
    data = get_data('data/iris.data')
    print(f'Fetched data of shape ({data.shape})', data)
