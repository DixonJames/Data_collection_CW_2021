from bs4 import BeautifulSoup as soup
from nltk.corpus import stopwords
from functools import cache
from itertools import chain

import pandas as pd
import random
import urllib3
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import itertools
from scipy import spatial
import math

import re
import nltk
import csv

import numpy
import scipy
import sklearn
import gensim
from collections import defaultdict

try:
    print("downloading common wordlists")
    nltk.download('stopwords')
    nltk.download('punkt')
except:
    print("couldn't download latest lists")

path = "distance.xlsx"

def tokenizeString(article):
    """
    removed common words and splits atricle into tokens in an array
    :param article: one string
    :return: list of words/tokens
    """
    import string
    blacklist = set({"'",'``', '\'\'' })
    s = set(stopwords.words('english'))
    article = article.replace(".", "")
    important_words = ""
    for word in article.split():
        if word not in s:
            important_words += (re.sub("\\\\", "", word) + " ")

    return [word.lower() for word in nltk.word_tokenize(important_words) if len(word) != 1 and (word not in blacklist) and word not in string.punctuation]

class FileErr(Exception):
    pass

class BBCextraction:


    def getPage(self, url, eaxtraHeaders):

        usrAgnt = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{random.randint(0, 1000)}.{random.randint(0, 50)} (KHTML, like Gecko) Chrome/{random.randint(10, 99)}.{random.randint(10, 99)}.{random.randint(1000, 9999)}.{random.randint(100, 999)} Safari/{random.randint(100, 999)}.{random.randint(10, 99)}"

        if eaxtraHeaders == None:
            headers = { 'User-Agent' : usrAgnt }
        else:
            headers = ({ 'User-Agent' : usrAgnt }).update(eaxtraHeaders)

        http = urllib3.PoolManager()
        response = http.request('GET', url, headers = headers)
        data = soup(response.data, "html.parser")

        #print(self.extractBBCLinks(data))
        return data

    def extractBBCLinks(self, data):
        links = []
        containers = data.findAll("div", {"class": "ssrcss-v4rel9-PromoContent e1f5wbog0"})
        for c in containers:
            url_section = c.findAll("div", {"class": "ssrcss-1uw1j0b-PromoHeadline e1f5wbog2"})
            if c.a["href"].split("/")[3] == "news":
                links.append(c.a["href"])

        return list(set(links))

    def searchPages(self):
        """
        goes though the XLM file and searches each item on bbc news and saves the resulting page in a text file
        :return:
        """
        keywords = fileMangagement().openXLS(path)['Keywords']
        testURL = "https://www.bbc.co.uk/search?q=durham"

        for word in list(keywords):
            seach_header = {'q': f"{word}"}
            content = self.getPage(testURL, seach_header)
            fileMangagement().saveTextFile(word, str(content), "base_searches")

    def BBCsearch(self, word):
        """
        produces all links to articles from a certain search on BBC
        :param word:
        :return:
        """
        links = []
        for p in range(30):
            url = f"https://www.bbc.co.uk/search?q={word}&page={p}"
            links.extend(self.extractBBCLinks(self.getPage(url, None)))

        return links

    def extractArticle(self, data):

        links = []
        try:
            title = str(data.findAll("h1", {"class": "ssrcss-1pl2zfy-StyledHeading e1fj1fc10"})[0].contents[0])
        except:
            title = str(data.findAll("h1")[0].contents[0])

        paragraphs = data.findAll("p")
        clean_paragraphs = [title]

        url_section = []

        for c in paragraphs:
            url_section = c.findAll("a", {"class": "ssrcss-hiczm3-InlineLink e1no5rhv0"})

            if url_section != []:
                if c.a["href"]:
                    links.append(c.a["href"])

            if len(c.findAll("a", {"class": "ssrcss-1hlxxic-PromoLink e1f5wbog6"})) == 0 and len(c.contents) != 0:
                if not ("class=" in str(c.contents[0])):
                    clean_paragraphs.append( re.sub("([\(\[]).*?([\)\]])", "\g<1>\g<2>", (re.sub("\\\\", "", str(c.contents[0]))).replace("\\", ""))
)

            article = ""

            for p in clean_paragraphs[:-1]:
                article += p

        return list(set(links)), article

class fileMangagement:

    def openXLS(self, path):
        try:
            keywords = pd.read_excel(path)
            return keywords
        except Exception as e:
            raise FileErr() from e

    def saveTextFile(self, name, contents, subfolderName=None, basefolder=None):
        import os.path

        PROJECT_ROOT_DIR = "."

        if basefolder == None:
            CHAPTER_ID = "BBC_pages"
        else:
            CHAPTER_ID = basefolder

        if subfolderName == None:
            save_path = os.path.join(PROJECT_ROOT_DIR, "data", CHAPTER_ID)
        else:
            save_path = os.path.join(PROJECT_ROOT_DIR, "data", CHAPTER_ID, subfolderName)

        os.makedirs(save_path, exist_ok=True)

        completeName = os.path.join(save_path, name + ".txt")

        with open(completeName, "w", encoding='utf-8') as file:
            try:
                file.write(contents)
            except:
                print("file write issue")


    def getContent(self, keywords):
        """
        goes though every keyword file that contains all its articles
        then created a dicts (with keys being the keywords):
        1. list of all articles separately
        2. list of all articles combined into one big string
        3. list of keywords (with blocking words removed)
        :param keywords:
        :return:
        """
        import os
        PROJECT_ROOT_DIR = "."


        #for each keyword makes one giant article
        articles_path = os.path.join(PROJECT_ROOT_DIR, "data", "Articles")

        directory = os.fsencode(articles_path)

        totalAritcles = {}
        seperateAritcles = {}
        for keyword in keywords:
            allArticels = ""
            seperateAllArticles = []

            for folder in os.listdir(directory):
                foldername = os.fsdecode(folder)

                if keyword in foldername:
                    for file_name in os.listdir(os.path.join(PROJECT_ROOT_DIR, "data", "Articles", foldername)):
                        if file_name.endswith(".txt"):
                            filename_str = os.fsdecode(file_name)
                            with open(os.path.join(PROJECT_ROOT_DIR, "data", "Articles", foldername, filename_str), "r", encoding="utf8") as f:
                                whole_article = f.read()
                                allArticels += re.sub("([\<(\[]).*?([\)\]\>])", "",(whole_article))
                                seperateAllArticles.append(re.sub("([\<(\[]).*?([\)\]\>])", "",(whole_article)))

            totalAritcles[f"{keyword}"] = allArticels
            seperateAritcles[f"{keyword}"] = seperateAllArticles


        words_path = os.path.join(PROJECT_ROOT_DIR, "data", "Prominent_words")

        totalDataFrames = {}
        for keyword in keywords:
            allArticels = ""

            for folder in os.listdir(words_path):
                foldername = os.fsdecode(folder)

                if keyword in foldername:
                    for file_name in os.listdir(os.path.join(PROJECT_ROOT_DIR, "data", "Prominent_words", foldername)):
                        if file_name.endswith(".csv"):
                            filename_str = os.fsdecode(file_name)

                            try:
                                totalDataFrames[f"{keyword}"] = pd.read_csv(os.path.join(PROJECT_ROOT_DIR, "data", "Prominent_words", foldername, filename_str))
                            except:
                                print("file open issue")



        return seperateAritcles, totalAritcles, totalDataFrames



    def saveCSVFile(self, name, contents, subfolderName = None, baseFolder = None):
        """
        saves a CSV file to the directory specified
        :param name:
        :param contents:
        :param subfolderName:
        :param baseFolder:
        :return:
        """
        import os.path

        PROJECT_ROOT_DIR = "."

        if baseFolder == None:
            CHAPTER_ID = "BBC_pages"
        else:
            CHAPTER_ID = baseFolder


        if subfolderName == None:
            save_path = os.path.join(PROJECT_ROOT_DIR, "data", CHAPTER_ID)
        else:
            save_path = os.path.join(PROJECT_ROOT_DIR, "data", CHAPTER_ID, subfolderName)

        os.makedirs(save_path, exist_ok=True)

        try:
            pd.DataFrame(contents).to_csv(os.path.join(save_path, f"{name}.csv",))
        except:
            print("file write issue")


    def downloadFile(self, url, CHAPTER_ID = None):
        """
        dowloads the resource at the specified URL to the file directory specified
        :param url:
        :param CHAPTER_ID:
        :return:
        """
        import requests
        import os.path

        PROJECT_ROOT_DIR = "."

        if CHAPTER_ID == None:
            CHAPTER_ID = "pretrainedModels"

        save_path = os.path.join("models", CHAPTER_ID)

        os.makedirs(save_path, exist_ok=True)

        r = requests.get(url)

        # open method to open a file on your system and write the contents
        with open("models/pretrainedModels/glove.6B.zip", "wb") as code:
            code.write(r.content)

class dataCollection:

    def keywordDataCollection(self):
        keywords = fileMangagement().openXLS(path)['Keywords']
        testURL = "https://www.bbc.co.uk/search"

        # testing
        # keywords = ["Durham"]
        wordnum = 0
        for word in list(keywords):
            wordnum += 1

            links = []
            prominent_words = []
            first_word = 0
            while len(links) < 100 and first_word <= len(word):
                s_parts = word.split(" ")[first_word:]
                searchword = ""
                for part in s_parts:
                    searchword += (part + " ")
                searchword = searchword[:-1]

                links.extend(BBCextraction().BBCsearch(searchword))
                links = list(set(links))
                first_word += 1

            i = 0
            for link in links:
                print(int((i/len(links))*100),"%, of:",wordnum, "/", len(list(keywords)), "keywords")
                i += 1

                content = BBCextraction().getPage(link, None)
                fileMangagement().saveTextFile(f"{word}_page_{i}", str(content), f"{word}_pages")

                externalLinks, article = BBCextraction().extractArticle(content)
                fileMangagement().saveTextFile(f"{word}_article_{i}", article, f"{word}_articles", "Articles")

                prominent_words.extend(tokenizeString(article))

            fileMangagement().saveCSVFile(f"{word}_prominent_words", list(set(prominent_words)), f"{word}", "Prominent_words")

class similarity:
    def __init__(self, keywords, articles):
        self.dict_keywords = keywords
        self.article = articles

        self.vectorisingDict = {}

    def getVecSources(self):
        """
        goes though the directory and checks that the necessary resources have been downloaded and unzipped.
        does downloads and unzips as necessary
        """

        from pathlib import Path
        import zipfile
        url = "http://nlp.stanford.edu/data/glove.6B.zip"

        # checks to see if has downloaded file

        my_file = Path("models/pretrainedModels/glove.6B.zip")
        if not(my_file.is_file()):
            print("File not accessible")
            print("Downloading resource file. This may take roughly 10 minutes depending on connection speeds!")
            fileMangagement().downloadFile(url, "pretrainedModels")

        #if not unzipped, unzips zip file
        my_file = Path("glove.6B.50d.txt")
        if not(my_file.is_file()):
            print("extracting ZIP")

            with zipfile.ZipFile("models/pretrainedModels/love.6B.zip", 'r') as zip_ref:
                zip_ref.extractall(".")

        my_file = Path("glove.6B.50d.txt")

        if  not(my_file.is_file()):
            print("files Acquired UNsuccessfully")





    @cache
    def vectoriseString(self, string):
        """
        uses the glove dataset to transform a word into a vector
        :param string: the string to be vecorised
        :return: vecotr as an array of ints
        """
        #checks to see if correct resources have been downloaded, and gets them if nessisary
        self.getVecSources()

        if self.vectorisingDict == {}:
            word_dict = {}
            with open("glove.6B.50d.txt", 'r', encoding="utf-8") as f:
                for line in f:
                    word_dict[line.split()[0]] = np.array(line.split()[1:], "float32")

            self.vectorisingDict = word_dict

        if string in self.vectorisingDict.keys():
            return self.vectorisingDict[string]
        else:
            return None



        #now should have required files!

    def jaccardSimilarity(self, setA_keywords, setB_keywords):
        """
        captain jaccard of the USS dataprise!
        :return: ratio of common keywords in sets
        """
        common = (setA_keywords).intersection(setB_keywords)
        union = (setA_keywords).union(setB_keywords)

        return len(common) / len(union)

    def avgVecKeywords(self, keywords, lengthVec):
        """
        finds the averave vector for a list of keywords
        :param keywords:
        :param lengthVec:
        :return:
        """
        a_vec = [0 for i in range(lengthVec)]
        for w in list(keywords):
            w_vec = self.vectoriseString(w)
            if not(w_vec is None):
                a_vec = np.add(a_vec, w_vec)

        for i in range(len(a_vec)):
            a_vec[i] = a_vec[i] / len(list(keywords))

        return a_vec

    def avgVector(self, listOfvectors):
        a_vec = [0 for i in range(len(listOfvectors[0]))]

        for w in listOfvectors:
            if not (w is None):
                a_vec = np.add(a_vec, w)

        for i in range(len(listOfvectors[0])):
            a_vec[i] = a_vec[i] / len(listOfvectors)

        return a_vec

    def vectoriseWord(self, word):
        return  self.avgVecKeywords(word, 50)

    def cosineSimilarity(self, setA_keywords, setB_keywords):
        from scipy import spatial
        a_vec = self.avgVecKeywords(setA_keywords["0"], 50)
        b_vec = self.avgVecKeywords(setB_keywords["0"], 50)

        similarity = 1 - spatial.distance.cosine(a_vec, b_vec)

        return similarity

class Doc2VecSimilarity:
    def __init__(self, corpus, articles):
        self.training_corpus = self.prePorcesCorpus(corpus)
        self.allArticles = articles

        self.model = self.doc2VecPipeline()

    def prePorcesCorpus(self, corpus):
        """
        docs = []
        train_corpus = []
        for kw in list(corpus):
            for article in corpus[kw]:
                docs.append(article)

        docs = list(set(docs))

        for i, line in enumerate(docs):
            tokens = gensim.utils.simple_preprocess(line)
            gensim.models.doc2vec.TaggedDocument(tokens, [i])
        """

        s = set(stopwords.words('english'))
        working_corpus = {}
        for key in list(corpus):
            articles = []
            for article in range(len(corpus[key])):
                articles.append(
                    [word for word in gensim.utils.simple_preprocess(corpus[key][article]) if not (word in s)])
            working_corpus[key] = articles

        # frequencey for words

        word_freq = defaultdict(int)
        for key in list(working_corpus):
            for aritcle in working_corpus[key]:
                for word in aritcle:
                    word_freq[word] += 1
        index = 0
        final_corpus = [[] for key in list(working_corpus)]
        for key in list(working_corpus):
            for aritcle in working_corpus[key]:
                final_corpus[index].append([word for word in aritcle if word_freq[word] > 1])
            index += 1

        return final_corpus

    def addTokens(self, doc, index):
        stringver = ""
        tokens = gensim.utils.simple_preprocess(' '.join(doc))
        return gensim.models.doc2vec.TaggedDocument(tokens, [index])

    def doc2VecVocab(self, corpai):
        training_corpai = []
        index = 0
        for doc in corpai:
            training_corpai.append(self.addTokens(doc, index))
            index += 1

        model = gensim.models.doc2vec.Doc2Vec(vector_size=50, min_count=2, epochs=40)
        model.build_vocab(training_corpai)
        return model, training_corpai


    def doc2VecPipeline(self):
        corpus = []
        for kw in self.training_corpus:
            for doc in kw:
                corpus.append(doc)
        corpus.sort()
        corpus = list(corpus for corpus, _ in itertools.groupby(corpus))

        model, training_corpai = self.doc2VecVocab(corpus)
        model.train(training_corpai, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    @cache
    def cosineSimilarity(self, a_vec, b_vec):
        similarity = 1 - spatial.distance.cosine(a_vec, b_vec)

        return similarity

    @cache
    def queryDoc2VecModel(self, articleString):
        return self.model.infer_vector(gensim.utils.simple_preprocess(articleString))





def jacSimDF(words_dict, article_dict, keywords):
    keywordlist = list(keywords)
    res = pd.DataFrame([[0.0 for i in range(len(keywordlist))] for i in range(len(keywordlist))], columns=keywordlist)

    allsets = [set(words_dict[list(words_dict)[i]]["0"]) for i in range(len(list(keywords)))]
    common = (allsets[0]).intersection(set(allsets[0]))
    for s in allsets:
        common = (common).intersection(s)

    for keywordA in list(keywords):
        for keywordB in list(keywords):
            word_index = keywordlist.index(keywordB)

            distance = similarity(words_dict, article_dict).jaccardSimilarity(set(words_dict[keywordA]["0"]) - common, set(words_dict[keywordB]["0"]) - common)
            if distance == 1.0:
                res[keywordA][word_index] = 1.0
            else:
                res[keywordA][word_index] = distance

    return res

def semanticSimDF(words_dict, article_dict, corpus,  keywords, justVecs = False):
    keywordlist = list(keywords)
    res = pd.DataFrame([[0.0 for i in range(len(keywordlist))] for i in range(len(keywordlist))], columns=keywordlist)
    workspace = similarity(words_dict, article_dict)

    if justVecs:
        labels = list(chain.from_iterable([[k_word for article in corpus[k_word]] for k_word in keywordlist]))
        vecs = list(chain.from_iterable([[workspace.avgVecKeywords(article, 50) for article in corpus[k_word]] for k_word in keywordlist]))
        return vecs, labels


    for keywordA in list(keywords):
        for keywordB in list(keywords):
            word_index = keywordlist.index(keywordB)

            distance = workspace.cosineSimilarity(words_dict[keywordA], words_dict[keywordB])

            res[keywordA][word_index] = distance

    return res

def Doc2VecDF(corpus, article_dict, keywords, justVecs = False):
    keywordlist = list(keywords)
    vector_res = pd.DataFrame([[0.0 for i in range(len(keywordlist))] for i in range(len(keywordlist))], columns=keywordlist)

    complexComparisonEngine = Doc2VecSimilarity(corpus, article_dict)

    if justVecs:
        labels = list(chain.from_iterable([[k_word for article in corpus[k_word]] for k_word in keywordlist]))
        vecs = list(chain.from_iterable([[complexComparisonEngine.queryDoc2VecModel(article) for article in corpus[k_word]] for k_word in keywordlist]))
        return vecs, labels

    #one vector per keyword
    #vectors = {article : complexComparisonEngine.queryDoc2VecModel(article_dict[article]) for article in keywordlist}
    #one vector per article
    vectors = {k_word: [complexComparisonEngine.queryDoc2VecModel(article) for article in corpus[k_word]] for k_word in keywordlist}



    for keywordA in list(keywords):
        index = 0
        for keywordB in list(keywords):
            vector_distances = 0


            if keywordB == keywordA:
                vector_res[keywordA][index] = 1.0
            else:
                for articleA in vectors[keywordA]:
                    for articleB in vectors[keywordB]:
                        vector_distances += 1 - spatial.distance.cosine(articleA, articleB)

                vector_res[keywordA][index] = vector_distances / (len(vectors[keywordA]) * (len(vectors[keywordB])))

            index += 1
    return vector_res



def normaliseDF(df):
    min, max = dfEdges(df)

    for col_name in df.columns:
        vals = df[f"{col_name}"]
        new_vals = []
        for val in vals:
            newval = (val - min)/(max - min)
            if newval > 1:
                newval = 1
            new_vals.append(newval)

        df[f"{col_name}"] = new_vals

    return df

def dfEdges(df):

    min = 1.0
    max = 0.0
    for col_name in df.columns:
        col_min = sorted(list(df[col_name]))[0]
        col_max = sorted(list(df[col_name]))[-2]

        if col_min < min:
            min = col_min
        if col_max > max:
            max = col_max

    return min, max

class visualisation:
    @staticmethod
    def heatMap(dataframe):

        dataframe.index = dataframe.columns
        ax = sns.heatmap(dataframe, vmin=0, vmax=1, annot=True, fmt="f",  linewidths=.5)

    @staticmethod
    def TwoDRepGraph(vectors, labels, mean = False):
        from sklearn.manifold import TSNE

        target = labels

        tsne = TSNE(n_components = 3)
        X_train_ = tsne.fit_transform(pd.DataFrame(vectors))

        df_graph = pd.DataFrame(X_train_, columns=['x', 'y'])
        df_graph['keyword'] = labels

        if mean:
            meanpoints = df_graph.groupby('keyword', as_index=False)['x'].mean()
            meanpoints['y'] = (df_graph.groupby('keyword', as_index=False)['y'].mean())['y']

            sns.scatterplot(meanpoints['x'], meanpoints['y'], hue=meanpoints['keyword'], palette='colorblind')


        sns.scatterplot(df_graph['x'], df_graph['y'], hue = df_graph['keyword'], palette='colorblind')



    @staticmethod
    def buildingGraphDF():
        #for doc2vec
        vecs, labels = Doc2VecDF(aritcles_corpus, article_dict, keywords, True)
        #for GloVe
        #vecs, labels = semanticSimDF(words_dict, article_dict, aritcles_corpus, keywords, True)

        visualisation.TwoDRepGraph(vecs, labels, True)

        df = pd.DataFrame(vecs)
        df['keyword'] = labels


    @staticmethod
    def averageDataFrame(dfList):
        collumbs = list(dfList[0])
        overall_mean_df = pd.DataFrame()

        for col in collumbs:
            coldf = pd.DataFrame()
            index = 0
            for df in dfList:
                index += 1
                coldf[f"{index}_col"] = df[col]


            all_cols = coldf.loc[:, f"1_col":f"{index}_col"]

            overall_mean_df[f'{col}'] = all_cols.mean(axis=1)

        return overall_mean_df

    @staticmethod
    def keywordExample(dfList, colname):
        #order = Jaccard, GloVe, Doc2Vec
        collumbs = list(dfList[0])
        overall_col = pd.DataFrame()

        overall_col["Jaccard"] = dfList[0][colname]
        overall_col["GloVe"] = dfList[1][colname]
        overall_col["Doc2Vec"] = dfList[2][colname]

        overall_col.index = overall.columns

        return overall_col







if __name__ == '__main__':
    #gets all data:

    """ pulls from the BBC
    dataCollection().keywordDataCollection()
    """
    #gets keywords form the Excell file
    keywords = fileMangagement().openXLS(path)['Keywords']
    aritcles_corpus, article_dict, words_dict = fileMangagement().getContent(keywords)




    """
    graphing
    """
    #s_df = semanticSimDF(words_dict, article_dict, aritcles_corpus, keywords, True)

    visualisation.buildingGraphDF()





    D2V_df = Doc2VecDF(aritcles_corpus, article_dict, keywords)

    j_df = jacSimDF(words_dict, article_dict, keywords)

    s_df = semanticSimDF(words_dict, article_dict, aritcles_corpus, keywords, False)

    j_df = normaliseDF(j_df)
    s_df = normaliseDF(s_df)
    D2V_df = normaliseDF(D2V_df)

    overall = visualisation.averageDataFrame([j_df, s_df, D2V_df])
    keyword = visualisation.keywordExample([j_df, s_df, D2V_df], 'DoS attack')



    #now normailse the DS
    visualisation.heatMap(D2V_df)
    plt.show()
    visualisation.heatMap(j_df)
    plt.show()
    visualisation.heatMap(s_df)
    plt.show()









