import csv

def get_data(filename=''):
    header = []
    data = []

    with open(filename) as csvfile:
        csvreader = csv.reader(csvfile)

        header = next(csvreader)

        for datapoint in csvreader:
            values = [value for value in datapoint]
            data.append(values)

    return data

if __name__ == '__main__':
    data = get_data('training_data/iris.data')
    print(data)
