import os
import collections
import re
import copy
import random
from nltk.stem import PorterStemmer 
import sys

def apply(weights, instance):
    weight_sum = weights['weight_zero']
    for i in instance:
        if i not in weights:
            weights[i] = 0.0
        weight_sum += weights[i] * instance[i]
    if weight_sum > 0:
        # return is spam
        return 1
    else:
        # return is ham
        return 0

def extractVocab(data_set,v):
    ps = PorterStemmer();
    for i in data_set:
        for j in i:
            if j not in v:
                v.append(ps.stem(j.lower()))
    return v

def removeStopWords(stops, data_set):
    filtered_data_set = copy.deepcopy(data_set)
    for i in stops:
        for j in filtered_data_set:
            if i in j:
                del j[i]
    return filtered_data_set

def setStopWords(stop_word_text_file):
    stops = []
    with open(stop_word_text_file, 'r') as txt:
        stops = (txt.read().splitlines())
    return stops


def bagOfWords(text):
    bagsofwords = collections.Counter(re.findall(r'\w+', text))
    return dict(bagsofwords)

def makeDataSet(class_list,directory):
    for dir_entry in os.listdir(directory):
        dir_entry_path = os.path.join(directory, dir_entry)
        if os.path.isfile(dir_entry_path):
            with open(dir_entry_path, 'r', encoding='ascii', errors='ignore') as text_file:
                text = text_file.read()
                class_list.append(bagOfWords(text))

def learnWeights(weights, eta, training_set, num_iterations):
    for i in num_iterations:
        for (key, value) in training_set.items():
            weight_sum = weights['weight_zero']
            for f in value:
                if f not in weights:
                    weights[f] = 0.0
                weight_sum += weights[f] * value[f]
                
            perceptron_output = 0.0
            if weight_sum > 0:
                perceptron_output = 1.0
                
            target_value = 0.0
            if "spam" in key:
                target_value = 1.0
            
            for w in value:
                weights[w] += float(eta) * float((target_value - perceptron_output)) *  float(value[w])
            
def main(train_dir, test_dir):
    
    print("")
    print("")
    print("Please wait till 20 combinations of eta and iterations are printed")
    print("")
    print("")
    print("")
    
    testingCombination = {}
    testingCombination.update({0.0087:22.6829})
    for i in range(19):
        testingCombination.update({random.uniform(0.001,0.05):random.uniform(10,100)})
    
    with open('result.txt','w') as f:
        f.write("")
        
    for (key, value) in testingCombination.items():
        
        iterations = str(round(value,4))
        learning_constant = str(round(key,4))
        
        train_spam = []
        train_ham = []
        test_spam = []
        test_ham =[]
    
        makeDataSet(train_spam,train_dir + "/spam")
        makeDataSet(train_ham,train_dir+"/ham")
        makeDataSet(test_spam,test_dir+"/spam")
        makeDataSet(test_ham,test_dir+"/ham")
    
        stop_words = setStopWords("stops.txt")

        filtered_train_spam = removeStopWords(stop_words, train_spam)
        filtered_train_ham = removeStopWords(stop_words, train_ham)
        filtered_test_spam = removeStopWords(stop_words, test_spam)
        filtered_test_ham = removeStopWords(stop_words, test_ham)
        
        filtered_train_vocab = []
        train_vocab = []
        
        extractVocab(train_spam,train_vocab)
        extractVocab(train_ham,train_vocab)
        
        extractVocab(filtered_train_spam,filtered_train_vocab)
        extractVocab(filtered_train_ham,filtered_train_vocab)
    
    
        filtered_weights = {'weight_zero': 1.0}
        for i in filtered_train_vocab:
            filtered_weights[i] = 0.0
    
        weights = {'weight_zero': 1.0}
        for i in train_vocab:
            weights[i] = 0.0
    
        
        filtered_emails = {}
        filter_counter = 1
        for i in filtered_train_spam:
            filtered_emails.update({"spam_" + str(filter_counter) : i})
            filter_counter = filter_counter + 1
        
        for i in filtered_train_ham:
            filtered_emails.update({"ham_" + str(filter_counter) : i})
            filter_counter = filter_counter + 1
        
        emails = {}
        counter = 1
        for i in train_spam:
            emails.update({"spam_" + str(counter) : i})
            counter = counter + 1
        
        for i in train_ham:
            emails.update({"ham_" + str(counter) : i})
            counter = counter + 1
   
     
        learnWeights(weights, learning_constant, emails, iterations)
        learnWeights(filtered_weights, learning_constant, filtered_emails, iterations)

        filter_num_correct_guesses = 0
        for i in filtered_test_spam:
            guess = apply(filtered_weights, i)
            if guess == 1:
                filter_num_correct_guesses += 1
        
        for i in filtered_test_ham:
            guess = apply(filtered_weights, i)
            if guess == 0:
                filter_num_correct_guesses += 1
            
        num_correct_guesses = 0
        for i in test_spam:
            guess = apply(weights, i)
            if guess == 1:
                num_correct_guesses += 1
        
        for i in test_ham:
            guess = apply(weights, i)
            if guess == 0:
                num_correct_guesses += 1
        
        print("")
        print("For eta value " + str(learning_constant) + " and interation " + str(iterations))
        print("")
        print("Emails guessed correctly: %d/%d" % (num_correct_guesses, len(test_spam)+ len(test_ham)))
        print("Accuracy with stopwords: %.4f%%" % (float(num_correct_guesses) / float(len(test_spam)+ len(test_ham)) * 100.0))
        print("")
        print("Filtered Emails guessed correctly: %d/%d" % (filter_num_correct_guesses, len(filtered_test_spam)+ len(filtered_test_ham)))
        print("Accuracy without stopwords: %.4f%%" % (float(filter_num_correct_guesses) / float(len(filtered_test_spam)+ len(filtered_test_ham)) * 100.0))
        print("")
        print("-----------------------------------------------")
        
        with open('result.txt','a') as f:
            f.write("\nFor eta value " + str(learning_constant) + " and interation " + str(iterations))
            f.write("\n")
            f.write("Emails guessed correctly: %d/%d" % (num_correct_guesses, len(test_spam)+ len(test_ham)))
            f.write("\nAccuracy with stopwords: %.4f%%" % (float(num_correct_guesses) / float(len(test_spam)+ len(test_ham)) * 100.0))
            f.write("\n\n")
            f.write("Filtered Emails guessed correctly: %d/%d" % (filter_num_correct_guesses, len(filtered_test_spam)+ len(filtered_test_ham)))
            f.write("\nAccuracy without stopwords: %.4f%%" % (float(filter_num_correct_guesses) / float(len(filtered_test_spam)+ len(filtered_test_ham)) * 100.0))
            f.write("\n\n")
            f.write("\n-----------------------------------------------")
            
    
    f.close()
    print("All the outputs available in result.txt file")

if __name__ == '__main__':
    main(sys.argv[1],sys.argv[2])