import os
import re
import string
import warnings

import PyPDF2
import nltk
import numpy as np
from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
from nltk.tokenize.toktok import ToktokTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from textblob import TextBlob

warnings.filterwarnings('ignore')


class DataProcessing:
    def __init__(self):
        self.docs = []
        # Tokenization of text
        self.tokenizer = ToktokTokenizer()
        # Setting English stopwords
        self.stopword_list = nltk.corpus.stopwords.words('english')
        self.wordnet_lemmatizer = WordNetLemmatizer()
        self.vectorizer = TfidfVectorizer()

    def tokenization(self):
        pass

    def strip_html(self, text):
        soup = BeautifulSoup(text, "html.parser")
        return soup.get_text()

    # Removing the square brackets
    def remove_between_square_brackets(self, text: string) -> string:
        return re.sub('\[[^]]*\]', '', text)

    def remove_special_characters(self, text: string) -> string:
        pattern = r'[^a-zA-z0-9\s]'
        text = re.sub(pattern, '', str(text))
        return text

    # Stemming the text
    def simple_stemmer(self, text: string) -> string:
        ps = nltk.porter.PorterStemmer()
        text2 = [ps.stem(word) for word in text.split()]
        text = ' '.join([ps.stem(word) for word in text.split()])
        return text

    # removing the stopwords
    def remove_stopwords(self, text: string, is_lower_case: bool = False) -> string:
        tokens = self.tokenizer.tokenize(text)
        tokens = [token.strip() for token in tokens]
        if is_lower_case:

            filtered_tokens = [token for token in tokens if token not in self.stopword_list and len(token) > 2]
        else:
            filtered_tokens = [token for token in tokens if token.lower() not in self.stopword_list and len(token) > 2]
        filtered_text = ' '.join(filtered_tokens)
        return filtered_text

    def text_normalize(self, text: string) -> string:
        term_word = []
        # spell checker
        for j in text:
            norm_spelling = TextBlob(j)
            term_word.append(str(norm_spelling.correct()))
        return term_word

    def lemmatizer(self, text: string) -> string:
        term_word2 = [self.wordnet_lemmatizer.lemmatize(d, pos='v') for d in text.split()]
        return term_word2

    # Removing the noisy text
    def denoise_text(self, text: string) -> string:
        text = self.strip_html(text)
        text = self.remove_between_square_brackets(text)
        text = self.remove_special_characters(text)
        text = self.simple_stemmer(text)
        text = self.remove_stopwords(text)
        text = self.lemmatizer(text)
        text = self.text_normalize(text)
        return text


class PositionList(DataProcessing):
    def __init__(self):
        super().__init__()

    def data_cleaning(self, docs: list) -> list:
        # Apply function on review column
        for i in range(0, len(docs)):
            text2 = []
            for j in range(0, len(docs[i])):
                data = self.denoise_text(str(docs[i][j]))
                text2.append(data)
            docs[i] = text2
        return docs

    def make_position_list(self, docs: list) -> dict:
        # Positional Index
        pos_index = {}
        fileno = 0
        for i in range(0, len(docs)):
            pos = 0
            page = 1
            for terms in docs[i]:
                for term in terms:
                    if term is not None:
                        pos = pos + 1
                        # If term already exists in the positional index dictionary.
                        if term in pos_index:

                            # Increment total freq by 1.
                            pos_index[term][0] = pos_index[term][0] + 1

                            # Check if the term has existed in that DocID before.
                            if fileno in pos_index[term][1]:
                                if page in pos_index[term][1][fileno]:
                                    pos_index[term][1][fileno][page].append(pos)
                                else:
                                    pos_index[term][1][fileno][page] = [pos]

                            else:
                                pos_index[term][1][fileno] = {}
                                pos_index[term][1][fileno][page] = [pos]

                        # If term does not exist in the positional index dictionary
                        # (first encounter).
                        else:

                            # Initialize the list.
                            pos_index[term] = []
                            # The total frequency is 1.
                            pos_index[term].append(1)
                            # The postings list is initially empty.
                            pos_index[term].append({})
                            # Add doc ID to postings list.
                            pos_index[term][1][fileno] = {}
                            pos_index[term][1][fileno][page] = [pos]

                page += 1
            fileno += 1
        return pos_index


class PDFReader(PositionList):
    def __init__(self):
        super().__init__()
        self.doc = []
        self.pages = []

    def read_pdf(self, files) -> list:
        # os.chdir("..")
        # print(os.getcwd())
        for i in files:
            # creating a pdf file object
            file_url = "." + i
            print("url: ", file_url)
            pdfFileObj = open(file_url, 'rb')
            print(pdfFileObj)
            # creating a pdf reader object
            pdfReader = PyPDF2.PdfFileReader(pdfFileObj)

            # printing number of pages in pdf file
            pg = pdfReader.numPages
            self.pages = []
            pages = self.read_pages(pg, pdfReader)
            self.doc.append(pages)
            # closing the pdf file object
            pdfFileObj.close()
        return self.doc

    def read_pages(self, pg: string, pdfReader: object) -> list:
        txt = ""
        # creating a page object
        for j in range(0, pg):
            pageObj = pdfReader.getPage(j)
            txt = pageObj.extractText()
            self.pages.append(np.char.replace(txt, ',', ''))
        return self.pages

    def check_position(self, docs: list) -> dict:
        docs = self.data_cleaning(docs)
        pos_index = self.make_position_list(docs)
        print(">>>>>>Waiting...............")
        print(len(pos_index))
        return pos_index

    def SearchQuery(self, pos_index: dict, query: string) -> list:
        print(">>>>> Query running........")
        try:
            sample_pos_idx = pos_index[str(query)]
        except Exception as e:
            print("Query Error : ", e)
            sample_pos_idx = []
        return sample_pos_idx
