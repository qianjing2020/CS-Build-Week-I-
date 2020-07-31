from utilities import Criterion

train = [
    ['Green', 'triagle', 2, 'Leaf'],
    ['Blue', 'polygon', 10, 'Sky'],
    ['Red', 'round', 8, 'Ballon'],
    ['Red', 'polygon', 1, 'Flower'],
    ['White', 'round', 1, 'Flower'],
    ['Green', 'polygon', 10, 'Meadow']
]
# Column labels.
header = ["color", "shape", "size", "label"]

criterion = Criterion(header[:-1], 0, 'Red')
for n, observation in enumerate(train):
    print(n, criterion)
    print(criterion.meet(observation))

# criterion = Criterion(header, 2, 9)
# for observation in train:
#     print(criterion)
#     print(criterion.meet(observation))
