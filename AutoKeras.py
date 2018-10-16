# Load MNIST data and AutoKeras library
from keras.datasets import mnist
from autokeras.classifier import ImageClassifier
# Set up test and training set
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape + (1,))
x_test = x_test.reshape(x_test.shape + (1,))

clf = ImageClassifier(verbose=True, searcher_args={'trainer_args':{'max_iter_num':5}})

clf.fit(x_train, y_train, time_limit=5 * 60 * 60)

clf.final_fit(x_train, y_train, x_test, y_test, retrain=False, trainer_args={'max_iter_num':10})

y = clf.evaluate(x_test, y_test)
print(y * 100)

best_model = clf.load_searcher().load_best_model()

print(best_model.n_layers)

from torchvision import models
print(best_model.produce_model())
