import nltk
import re
from nltk.corpus import stopwords,wordnet
import unicodedata
import string
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer, LancasterStemmer

stopwords_list=stopwords.words('english')

# data=[list of all sentences]

porter = PorterStemmer()
lancaster=LancasterStemmer()
lemmatizer = WordNetLemmatizer()

def unicode_to_ascii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}

    return tag_dict.get(tag,wordnet.NOUN)

def clean_data(w):

    w = unicode_to_ascii(w)
    w=w.lower()                        # Lower casing
    w=re.sub(' +', ' ', w).strip(' ')  # Remove multiple whitespaces, also leading and trailing whitespaces
    w=re.sub(r'[^\w\s]','',w)          # Remove special characters and punctuation
    w=re.sub(r"([0-9])", r" ",w)       # Remove Numerical data
    w=re.sub("(.)\\1{2,}", "\\1", w)   # Remove duplicate characters
    words = w.split()                  # Tokenization
    clean_words = [word for word in words if (word not in stopwords_list) and len(word) > 2]
#     clean_words= [lemmatizer.lemmatize(w,get_wordnet_pos(w)) for w in clean_words] # For lemmatization (generally not used)
#     clean_words=[porter.stem(word) for word in clean_words]                        # For Porter stemmer
#     clean_words=[lancaster.stem(word) for word in clean_words]                     # For lancaster stemmer
    return " ".join(clean_words)

preprocessed_data=list(map(clean_data,data))

for i in range(len(preprocessed_data)):
    print(i+1,"Original text: \n{}\n\nPreprocessed text: \n{}\n".format(data[i],preprocessed_data[i]))
