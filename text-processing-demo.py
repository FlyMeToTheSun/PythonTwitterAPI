import spacy
import nltk
import re
from nltk.corpus import stopwords
import unicodedata
import string

# Punctuation removal
string_no_punct=re.sub(r'[^\w\s]','',string_inp)
print('String input:\n{}\n\nString input with no punctuations:\n{}'.format(string_inp,string_no_punct))

# Stop words removal
string_inp='Hey, @all do youuuuu want to learn Natural Language Processinggg 100% ??'
stopwords_list=stopwords.words('english')
string_no_stopwords = [word for word in string_inp.split() if (word not in stopwords_list) and len(word) > 2]
string_no_stopwords=" ".join(string_no_stopwords)
print('String with stopwords: {} \nString without stopwords and short words: {}'.format(string_inp,string_no_stopwords))

# Removing numerical data from the text
string_no_num=re.sub(r’[0–9]’,’’,string_inp)

# Removing multiple whitespaces in the text
re.sub(' +', ' ',string_inp)

# Removing duplicate characters in a word
re.sub("(.)\\1{2,}", "\\1", 'string_inp')

# Tokenization
from nltk.tokenize import word_tokenize

string_inp='hey you want learn natural language processing'
words = word_tokenize(string_inp)
tokens=string_inp.split()

# Sentence tokenization:
from nltk.tokenize import sent_tokenize
text = "hey you want to learn natural language processing! let's go"
output_text=sent_tokenize(text)

# Lemmatization
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize(word,pos))

# Stemming
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer

porter = PorterStemmer()
lancaster=LancasterStemmer()

print(porter.stem(word))
print(lancaster.stem(word))
