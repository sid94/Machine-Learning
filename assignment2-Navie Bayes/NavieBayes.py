import sys
import os
import string
from nltk.stem import PorterStemmer 
import math

def trainNavieBayes(classification,traindata,stopfile=None):
    if(stopfile != None):
        stopkeys = readingstopfile(stopfile)
    uniqueVocabulary, docCountinBothClass, docCountinEachClass, allWordsofEachClass = processingDocs(traindata)

    #removing stopwords from the unique vocab array
    if(stopfile != None):
        uniqueVocabulary = [x for x in uniqueVocabulary if x not in stopkeys]
        
    uniqueVocabularylength = len(uniqueVocabulary);
    
    #priors and conditional probability
    priors = []
    spamEachWordCountDict = {}
    hamEachWordCountDict = {}
    spamCondProbDict={}
    hamCondProbDict={}
    for i in range (classification):
        priors.append(docCountinEachClass[i] / float(docCountinBothClass))
        allTextEachClass = allWordsofEachClass[i]
        
        #remove stop keeys form all words of each class
        if(stopfile != None):
            allTextEachClass = [x for x in allTextEachClass if x not in stopkeys]
        
        denominatorForSpam = 0
        denominatorForHam = 0
#        totalcountinspamham = allTextEachClass[0] + 
        if(i==0):
            denominatorForSpam = uniqueVocabularylength + len(allTextEachClass)
        else:
            denominatorForHam = uniqueVocabularylength + len(allTextEachClass)
        
        for j in range (len(uniqueVocabulary)):
            if(i == 0):
               spamEachWordCountDict[uniqueVocabulary[j]] = allTextEachClass.count(uniqueVocabulary[j])
            else:
               hamEachWordCountDict[uniqueVocabulary[j]] = allTextEachClass.count(uniqueVocabulary[j])
    
        for k in range (len(uniqueVocabulary)):
            if(i == 0):
                val = (spamEachWordCountDict[uniqueVocabulary[k]]+ 1)/ float(denominatorForSpam)
                spamCondProbDict[str(uniqueVocabulary[k])] = val
            else:
                val = (hamEachWordCountDict[uniqueVocabulary[k]]+ 1)/ float(denominatorForHam)
                hamCondProbDict[str(uniqueVocabulary[k])] = val
    
    return uniqueVocabulary,priors,spamCondProbDict,hamCondProbDict
            
            
def readingstopfile(stopfile):
    stopwords = []
    filePath = stopfile
    with open(filePath,'r', encoding = 'raw_unicode_escape') as f:
            for line in f:
                line = line.translate(str.maketrans('','',string.punctuation))
                for word in line.split():
                    stopwords.append(word.lower())      
    return stopwords

def processingDocs(d):
    docCountinEachClass = []    # [doc in spam,doc in ham]
    allWordsofEachClass = []    #all the words from spam and ham
    uniqueVocabulary =[]        #unique from spam and ham
    docCountinBothClass = 0     #total no of docs in spam and ham
    ps = PorterStemmer();
    for i in range (len(d)):
        allWordsofEachClass.append([])
        dlist = os.listdir(d[i])
        docCountinEachClass.append(len(dlist))
        docCountinBothClass = docCountinBothClass + len(dlist)
        for j in range (docCountinEachClass[i]):
            filePath = d[i] + dlist[j]
            with open(filePath,mode = 'r',encoding = 'raw_unicode_escape') as f:
                for line in f:
                    line = line.translate(str.maketrans('','',string.punctuation))
                    for word in line.split():
                        allWordsofEachClass[i].append(ps.stem(word.lower()))
                        uniqueVocabulary.append(ps.stem(word.lower()))
    uniqueVocabulary = list(set(uniqueVocabulary)) #making words unique
    uniqueVocabulary.sort()
    return uniqueVocabulary, docCountinBothClass, docCountinEachClass, allWordsofEachClass

def NavieBayesTrainandTest(classification, uniqueVocabulary, priors, spamCondProbDict, hamCondProbDict, vocabularyinTestData):
    spamScore = 0
    hamScore = 0
    arr = []
    for i in range(len(vocabularyinTestData)):
        if(vocabularyinTestData[i] in uniqueVocabulary):
            arr.append(vocabularyinTestData[i])
    
    for i in range (classification):
        if(i == 0):
            spamScore = math.log(priors[i], 2)
        else:
            hamScore = math.log(priors[i], 2)
        for j in range(len(arr)):
            if(i == 0):
                spamScore = spamScore + math.log(spamCondProbDict[str(arr[j])], 2) 
            else:
                hamScore = hamScore + math.log(hamCondProbDict[str(arr[j])], 2)
    value = 0
    if(spamScore > hamScore): 
        value = 0
    else:
        value = 1
    return value

def MultinomialnavieBayes(uniqueVocabulary,priors,spamCondProbDict,hamCondProbDict,testdata):
    prediction = 0
    docCountInTestEachClass = len(os.listdir(testdata[1])) + len(os.listdir(testdata[0]))
    for i in range(len(testdata)):
        dlist = os.listdir(testdata[i])
        for j in range (len(dlist)):
            filePath = testdata[i] + dlist[j]
            with open(filePath,mode = 'r',encoding = 'raw_unicode_escape') as f:
                vocabularyinTestData = []
                for line in f:
                    line = line.translate(str.maketrans('','',string.punctuation))
                    for word in line.split():
                        vocabularyinTestData.append(word)
            index = NavieBayesTrainandTest(classification, uniqueVocabulary, priors, spamCondProbDict, hamCondProbDict, vocabularyinTestData)
            if(index == i):
                prediction = prediction + 1
    print(prediction)
    print("\nAccuracy: ", (prediction * 100) / float(docCountInTestEachClass))
    with open('result.txt','a') as f:
        f.write("\nAccuracy: " + str((prediction * 100) / float(docCountInTestEachClass)))
    

#Main
    
if __name__ == '__main__':
    if(len(sys.argv) != 4):
        print("please enter :python NavieBayes.py train test stops.txt")
        sys.exit()
    
    classification = 2 #spam and ham
    traindata = [];    #store paths of train docs
    testdata = [];     #store paths of test docs

    traindata.append(sys.argv[1] + "/spam/");
    traindata.append(sys.argv[1] + "/ham/");
    testdata.append(sys.argv[2] + "/spam/");
    testdata.append(sys.argv[2] + "/ham/");
    
    with open('result.txt','w') as f:
        f.write("Navie Bayes Output file")
    
    print("Accuracy with stopwords")
    with open('result.txt','a') as f:
        f.write("\nAccuracy with stopwords")
#        print("Accuracy with stopwords",file=f)
    uniqueVocabulary,priors,spamCondProbDict,hamCondProbDict = trainNavieBayes(classification,traindata)
    MultinomialnavieBayes(uniqueVocabulary,priors,spamCondProbDict,hamCondProbDict,testdata)
    
    print("\nAccuracy without stopwords")
    with open('result.txt','a') as f:
        f.write("\nAccuracy without stopwords")
#        print("\nAccuracy without stopwords",file=f)
    uniqueVocabulary,priors,spamCondProbDict,hamCondProbDict = trainNavieBayes(classification,traindata,sys.argv[3])
    MultinomialnavieBayes(uniqueVocabulary,priors,spamCondProbDict,hamCondProbDict,testdata)

    print("\nAccuracy on train data")
    with open('result.txt','a') as f:
        f.write("\nAccuracy on train data")
    MultinomialnavieBayes(uniqueVocabulary,priors,spamCondProbDict,hamCondProbDict,traindata)
    
    f.close()
    






    