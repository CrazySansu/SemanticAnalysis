import os
from nltk.tokenize import word_tokenize
from nltk.classify.util import accuracy
from nltk.classify import NaiveBayesClassifier
import pickle

def format_sentence(sentence):
	return {word:True for word in word_tokenize(sentence)}

def save_classifier(classifier):
   f = open('algo.pickle', 'wb')
   pickle.dump(classifier, f, -1)
   f.close()

train_pos_path1 = "trainingData/train/pos_from_TripAdvisor"
train_neg_path1 = "trainingData/train/neg_from_TripAdvisor"

test_pos_path1 = "trainingData/test/pos_from_TripAdvisor"
test_neg_path1 = "trainingData/test/neg_from_TripAdvisor"

print('Training model, please wait...')

training_data = []
for file in os.listdir(train_pos_path1):
        current = os.path.join(train_pos_path1, file)
        if os.path.isfile(current):
            data = open(current, "r").read()
            training_data.append([format_sentence(data),'positive'])

for file in os.listdir(train_neg_path1):
        current = os.path.join(train_neg_path1, file)
        if os.path.isfile(current):
            data = open(current, "r").read()
            training_data.append([format_sentence(data),'negative'])
print("50% done...")
testing_data = []
for file in os.listdir(test_pos_path1):
        current = os.path.join(test_pos_path1, file)
        if os.path.isfile(current):
            data = open(current, "r").read()
            testing_data.append([format_sentence(data),'positive'])

for file in os.listdir(test_neg_path1):
        current = os.path.join(test_neg_path1, file)
        if os.path.isfile(current):
            data = open(current, "r").read()
            testing_data.append([format_sentence(data),'negative'])
print("Completed!!")

bayesModel = NaiveBayesClassifier.train(training_data)
print('Saving model, please wait...')
save_classifier(bayesModel)
print('Model Saved! \n')
print('Testing Accuracy, please wait...')
acc =(accuracy(bayesModel,testing_data) * 100)
#str(Decimal(acc).quantize(TWOPLACES))
acc = '{0:.2f}'.format(acc)
print('Accuracy: ' + str(acc) + '%')
