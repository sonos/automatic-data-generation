from automatic_data_generation.utils.io import read_csv, write_csv
from pathlib import Path
from sklearn.model_selection import StratifiedShuffleSplit

data = read_csv(Path('data.csv'))
header = [data[0]]
data = data[1:]

intents = [row[3] for row in data]

test_fraction = 0.2
test_size = int(test_fraction * len(data))

sss = StratifiedShuffleSplit(n_splits=1,
                             test_size=test_size)

train_indices, test_indices = list(sss.split(intents, intents))[0]

train = header + [data[i] for i in train_indices]
validate = header + [data[i] for i in test_indices]

write_csv(validate, Path('validate.csv'))
write_csv(train, Path('train.csv'))

