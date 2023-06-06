# Requried imports
import os
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer, sent_tokenize
import numpy as np
from goose3 import Goose

#File locations of all the required files needed

inputData = 'D:\Data Science\Text Analysis\Input.xlsx' 
positiveWords = 'D:/Data Science/Text Analysis\MasterDictionary/positive-words.txt'
negativeWords = 'D:/Data Science/Text Analysis\MasterDictionary/negative-words.txt'
stopWordsAuditor = 'D:/Data Science/Text Analysis/StopWords/StopWords_Auditor.txt'
stopWordsCurrencies = 'D:/Data Science/Text Analysis/StopWords/StopWords_Currencies.txt'
stopWordsDatesAndNum = 'D:/Data Science/Text Analysis/StopWords/StopWords_DatesandNumbers.txt'
stopWordsGenric = 'D:/Data Science/Text Analysis/StopWords/StopWords_Generic.txt'
stopWordsGenericLong = 'D:/Data Science/Text Analysis/StopWords/StopWords_GenericLong.txt'
stopWordsGeographic = 'D:/Data Science/Text Analysis/StopWords/StopWords_Geographic.txt'
stopWordsNames = 'D:/Data Science/Text Analysis/StopWords/StopWords_Names.txt'\

# function to fetch Articles from the given links .

def getArticles(dataframe, URL):
    p = [] 
    for i in range(dataframe.shape[0]):
        url = dataframe.URL[i]
        article = Goose().extract(url=url)
        article = str(article.cleaned_text)
        p.append(article)

    dataframe['ARTICLE'] = p

# function to fetch Articles from the given links .

def getArticles(dataframe, URL):
    p = [] 
    for i in range(df.shape[0]):
        url = dataframe.URL[i]
        article = Goose().extract(url=url)
        article = str(article.cleaned_text)
        p.append(article)

    dataframe['ARTICLE'] = p

## Loading the stopWord files to a list.

with open(stopWordsAuditor ,'r') as stop_words_auditor:
    stop_words_auditor = stop_words_auditor.read().lower()
stop_words_auditor = stop_words_auditor.split('\n')
stop_words_auditor[-1:] = []

with open(stopWordsCurrencies ,'r') as stop_words_curren:
    stop_words_curren = stop_words_curren.read().lower()
stop_words_curren = stop_words_curren.split('\n')
stop_words_curren[-1:] = []

with open(stopWordsDatesAndNum ,'r') as stop_words_DateAndNum:
    stop_words_DateAndNum = stop_words_DateAndNum.read().lower()
stop_words_DateAndNum = stop_words_DateAndNum.split('\n')
stop_words_DateAndNum[-1:] = []

with open(stopWordsGenric ,'r') as stop_words_generic:
    stop_words_generic = stop_words_generic.read().lower()
stop_words_generic = stop_words_generic.split('\n')
stop_words_generic[-1:] = []

with open(stopWordsGenericLong ,'r') as stop_words_genericLong:
    stop_words_genericLong = stop_words_genericLong.read().lower()
stop_words_genericLong = stop_words_genericLong.split('\n')
stop_words_genericLong[-1:] = []

with open(stopWordsGeographic ,'r') as stop_words_geographic:
    stop_words_geographic = stop_words_geographic.read().lower()
stop_words_geographic = stop_words_geographic.split('\n')
stop_words_geographic[-1:] = []

with open(stopWordsNames ,'r') as stop_words_names:
    stop_words_names = stop_words_names.read().lower()
stop_words_names = stop_words_names.split('\n')
stop_words_names[-1:] = []

# Combining alll the stopwords list to a single list.

final_list = stop_words_names+stop_words_geographic+stop_words_auditor+stop_words_curren+stop_words_DateAndNum+stop_words_generic+stop_words_genericLong 

# Tokenizer : to filter out all the stop words.
def tokenizer(text):
    text = text.lower()
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)
    filtered_words = list(filter(lambda token: token not in final_list, tokens))
    return filtered_words

# Loading positive words to a list
with open(positiveWords,'r') as posfile:
    positiveWords=posfile.read().lower()
positiveWordList=positiveWords.split('\n')

# Loading negative words to a list
with open(negativeWords ,'r') as negfile:
    negativeWords=negfile.read().lower()
negativeWordList=negativeWords.split('\n')

# Calculating positive score 
def positive_score(text):
    numPosWords = 0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in positiveWordList:
            numPosWords  += 1
    
    sumPos = numPosWords
    return sumPos

# Calculating Negative score
def negative_score(text):
    numNegWords=0
    rawToken = tokenizer(text)
    for word in rawToken:
        if word in negativeWordList:
            numNegWords -=1
    sumNeg = numNegWords 
    sumNeg = sumNeg * -1
    return sumNeg


# Calculating polarity score

from textblob import TextBlob

def polarity_score(text):
    return TextBlob(text).sentiment.polarity

# Calculating Subjectivity score
def subjectivity_score(text):
    return TextBlob(text).sentiment.polarity

# Syllablle per Word

from nltk.corpus import cmudict
import nltk
nltk.download('cmudict')
nltk.download('punkt')
d = cmudict.dict()

    # Calculating syllablle in a input text.
def syllables(text):
    count = 0
    vowels = 'aeiouy'
    text = text.lower()
    if text[0] in vowels:
        count +=1
    for index in range(1,len(text)):
        if text[index] in vowels and text[index-1] not in vowels:
            count +=1
    if text.endswith('e'):
        count -= 1
    if text.endswith('le'):
        count+=1
    if count == 0:
        count +=1
    return count

    # number-of-syllables-in-a-word
def nsyl(word):

    try:
        return max([len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]])
    except KeyError:
        #if word not found in cmudict
        return syllables(word)


# Calculating Average sentence length 
# It will calculated using formula --- Average Sentence Length = the number of words / the number of sentences
def average_sentence_length(text):
    sentence_list = sent_tokenize(text)
    tokens = tokenizer(text)
    totalWordCount = len(tokens)
    totalSentences = len(sentence_list)
    average_sent = 0
    if totalSentences != 0:
        average_sent = totalWordCount / totalSentences
    
    average_sent_length = average_sent
    
    return round(average_sent_length)

# Calcutating Average Number of Words per Sentence..
# It is calculated using formula = sum_of_words in sentence / num_of _entences
def avg_num_words_per_sentence(text):
    parts = [len(l.split()) for l in re.split(r'[?!.]', text) if l.strip()]
    return (sum(parts)/len(parts))

# Calcutating Average Word length.
# It is calculated using the formula = total length of all words / total num of words..
def average_word_length(text):
    filtered = ''.join(filter(lambda x: x not in '".,;!-()', text))
    words = [word for word in filtered.split() if word]
    avg = sum(map(len, words))/len(words)
    return avg

# Counting complex words
def complex_word_count(text):
    tokens = tokenizer(text)
    complex_word_count = 0
    
    for word in tokens:
        if nsyl(word.lower()) > 2:
                    complex_word_count += 1
    return complex_word_count

# Calculating percentage of complex word 
# It is calculated using Percentage of Complex words = the number of complex words / the number of words 

def percentage_complex_word(text):
    tokens = tokenizer(text)
    complex_word_count = 0
    complex_word_percentage = 0
    
    for word in tokens:
        if nsyl(word.lower()) > 2:
                    complex_word_count += 1
    if len(tokens) != 0:
        complex_word_percentage = complex_word_count/len(tokens)
    
    return complex_word_percentage


# calculating Fog Index 
# Fog index is calculated using -- Fog Index = 0.4 * (Average Sentence Length + Percentage of Complex words)
def fog_index(averageSentenceLength, percentageComplexWord):
    fogIndex = 0.4 * (averageSentenceLength + percentageComplexWord)
    return fogIndex

#Counting total words
def total_word_count(text):
    tokens = tokenizer(text)
    return len(tokens)

# Caculating the pronouns presenet in the text. 
def personal_pronoun(text):
    pronounRegex = re.compile(r'\b(I|we|my|ours|(?-i:us))\b',re.I)
    pronouns = pronounRegex.findall(text)
    # to get all pronouns preseny use only `return pronouns`
    return len(pronouns)

# Loading the dataset
df = pd.read_excel(inputData)

# Removing out the unused columns 
df=df[['URL_ID','URL']]

# Getting the article from the url
getArticles(df,df.URL)

# Applying the abpve made fucntions
df['positive_score'] = df['ARTICLE'].apply(lambda x : positive_score(x))
df['negative_score'] = df['ARTICLE'].apply(lambda x : negative_score(x))
df['polarity_score'] = df['ARTICLE'].apply(lambda x: polarity_score(x))
df['subjectivity_score'] = df['ARTICLE'].apply(lambda x: subjectivity_score(x))
df['average_sentence_length'] = df['ARTICLE'].apply(lambda x : average_sentence_length(x))
df['percentage_of_complex_words'] = df['ARTICLE'].apply(lambda x : percentage_complex_word(x))
df['fog_index'] = df['ARTICLE'].apply(lambda x :fog_index(average_sentence_length(x),percentage_complex_word(x)))
df['avg_word_per_sentence']= df['ARTICLE'].apply(lambda x : avg_num_words_per_sentence(x))
df['complex_word_count']= df['ARTICLE'].apply(lambda x : complex_word_count(x))
df['word_count'] = df['ARTICLE'].apply(lambda x : total_word_count(x))
df['syllable_per_word'] = df['ARTICLE'].apply(lambda x : nsyl(x))
df['num_of_personal_pronouns'] = df['ARTICLE'].apply(lambda x : personal_pronoun(x))
df['average_word_length'] = df['ARTICLE'].apply(lambda x : average_word_length(x))

# Drooping the Article Column extracted 
df.drop('ARTICLE', inplace=True, axis=1)

df.to_csv('output.csv')