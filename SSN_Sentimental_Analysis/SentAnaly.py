import nltk
import os
import pickle
import time

def load_classifier():
                file = open('algo.pickle', 'rb')
                classifier = pickle.load(file)
                file.close()
                return classifier

def format_sentence(sent):
	return {word:True for word in nltk.word_tokenize(sent)}

def extract_Enames(sentence):                       #recursive function to identify entity names
    entity_names = []
    if hasattr(sentence, 'label') and sentence.label:
        if sentence.label() == 'NE':
            entity_names.append(' '.join([child[0] for child in sentence]))
        else:
            for child in sentence:
                entity_names.extend(extract_Enames(child))
    return entity_names

def intersect(a, b):
    return (set(a).intersection(b))

train_pos_path = "trainingData/train/pos_from_TripAdvisor"
train_neg_path = "trainingData/train/neg_from_TripAdvisor"

test_pos_path = "trainingData/test/pos_from_TripAdvisor"
test_neg_path = "trainingData/test/neg_from_TripAdvisor"

pos_words = "trainingData/Words/Pos"
neg_words = "trainingData/Words/Neg"
pos=[]
neg=[]
all_positive=[]
all_negative=[]
for file in os.listdir(pos_words):
        current = os.path.join(pos_words, file)
        if os.path.isfile(current):
            data = open(current, "r").read()
            pos.append(format_sentence(data))

for file in os.listdir(neg_words):
        current = os.path.join(neg_words, file)
        if os.path.isfile(current):
            data = open(current, "r").read()
            neg.append(format_sentence(data))
for word in pos:     
    for key in word:
        all_positive.append(key)

for word in neg:
    for key in word:
        all_negative.append(key)

bayesModel = load_classifier()                      #Object to call method that loads the classifier file
while(True):
    review = input("\nReview: ")
    print("")

    entity_names = []
    
    Npos = 0;
    Nneg = 0;
    positive_words = []
    negative_words = []
    sentences = nltk.sent_tokenize(review)
    all_words=[]
    for sent in sentences:                          #For each sentence in reveiw
        words = nltk.word_tokenize(sent)            #Seperate each word of sentence
        all_words.extend(words)
        ptag = nltk.pos_tag(words)                  #Give each word a POS Tag
        Ename = nltk.ne_chunk(ptag,binary=True)     #Use Chunk to find Entity name
        entity_names.extend(extract_Enames(Ename))
        rev = str(bayesModel.classify(format_sentence(sent)))
        print(rev)
        if(rev == "positive"):
            positive_words.append(sent)
            Npos = Npos + 1
        else:
            negative_words.append(sent)
            Nneg = Nneg + 1
    
    var=None
    if(Npos > Nneg):
        var = "Positive"
        print("Given review is Positive.")
    else:
        if(Nneg > Npos):
            var = "Negative"
            print("Given review is Negative.")
        else:
            var = "Neutral"
            print("Given review is Neutral.")

    if(len(entity_names) > 0):
        result = "Entities are: " + "".join(str(word)+', ' for word in set(entity_names))
        new_result = list(result)
        new_result[len(result) - 2 ] = "."
        print("".join(new_result))
        
        if(var == "Positive"):
            # print("Positive words are: ",positive_words)
            print("Positive words are: ",intersect(all_positive, all_words))
        if(var == "Negative"):
            #print("Negative words are: ",positive_words) 
            print("Negative words are: ",intersect(all_negative, all_words))
        if(var == "Neutral"):
            #print("Negative words are: ",positive_words) 
            print("Negative words are: ",intersect(all_negative, all_words))
            # print("Positive words are: ",positive_words)
            print("Positive words are: ",intersect(all_positive, all_words))
        
    else:
            print('No entity found!!')

    valid = input("\nDo you agree with our review? (y/n): ")
    if valid == 'y' or valid == 'Y':                    #Create a new file for trainig set
        print("This review has been added to our Training set!")
        moment=time.strftime("%Y-%b-%d__%H_%M_%S",time.localtime())
        if str(var)== 'Positive' :
            newF = open(os.path.join(train_pos_path,'output'+moment+'.txt'), 'w')
            newF.write(review+'\n')
            newF.close()
        if str(var) == 'Negative':
            newF = open(os.path.join(train_neg_path,'output'+moment+'.txt'), 'w')
            newF.write(review+'\n')
            newF.close()
            
    answer = input("\nCheck another review(y/n): ")
    if(answer == "y" or answer == "Y"):
        print("")
    else:
        print("Thank you for your feedback!")
        break
