from bs4 import BeautifulSoup as soup
from nltk.corpus import stopwords
from functools import cache
from itertools import chain, groupby

import pandas as pd
import random
import urllib3
import numpy as np

import seaborn as sns
import matplotlib.pyplot as plt


from pathlib import Path
import zipfile
from scipy import spatial
from sklearn.manifold import TSNE


import re
import nltk

import gensim
import string
import os



import requests

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

    blacklist = set({"'", '``', '\'\''})
    s = set(stopwords.words('english'))
    article = article.replace(".", "")
    important_words = ""
    for word in article.split():
        if word not in s:
            important_words += (re.sub("\\\\", "", word) + " ")

    return [word.lower() for word in nltk.word_tokenize(important_words) if
            len(word) != 1 and (word not in blacklist) and word not in string.punctuation]


class FileErr(Exception):
    pass


class BBCextraction:

    def getPage(self, url, eaxtraHeaders):

        usrAgnt = f"Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/{random.randint(0, 1000)}.{random.randint(0, 50)} (KHTML, like Gecko) Chrome/{random.randint(10, 99)}.{random.randint(10, 99)}.{random.randint(1000, 9999)}.{random.randint(100, 999)} Safari/{random.randint(100, 999)}.{random.randint(10, 99)}"

        if eaxtraHeaders == None:
            headers = {'User-Agent': usrAgnt}
        else:
            headers = ({'User-Agent': usrAgnt}).update(eaxtraHeaders)

        http = urllib3.PoolManager()
        response = http.request('GET', url, headers=headers)
        data = soup(response.data, "html.parser")

        # print(self.extractBBCLinks(data))
        return data

    def extractBBCLinks(self, data):
        links = []
        containers = data.findAll("div", {"class": "ssrcss-1q56g2r-PromoPortrait-PromoSwitchLayoutAtBreakpoints e3z3r3u0"})
        for c in containers:
            url_section = c.findAll("div", {"class": "ssrcss-1bn8j6y-PromoContent e1f5wbog0"})
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

            #promo articles on the side of news page
            if len(c.findAll("a", {"class": "ssrcss-1av146u-PromoHeadline e1f5wbog1"})) == 0 and len(c.contents) != 0:

                clean_paragraphs.append(re.sub('<[^>]+>', '',re.sub("<([\(\[]).*?([\)\]])>", "\g<1>\g<2>",(re.sub("\\\\", "", str(c.contents[0]))).replace("\\", ""))))

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

        PROJECT_ROOT_DIR = "."

        # for each keyword makes one giant article
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
                            with open(os.path.join(PROJECT_ROOT_DIR, "data", "Articles", foldername, filename_str), "r",
                                      encoding="utf8") as f:
                                whole_article = f.read()
                                allArticels += re.sub("([\<(\[]).*?([\)\]\>])", "", (whole_article))
                                seperateAllArticles.append(re.sub("([\<(\[]).*?([\)\]\>])", "", (whole_article)))

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
                                totalDataFrames[f"{keyword}"] = pd.read_csv(
                                    os.path.join(PROJECT_ROOT_DIR, "data", "Prominent_words", foldername, filename_str))
                            except:
                                print("file open issue")

        return seperateAritcles, totalAritcles, totalDataFrames

    def openSavedPage(self, keyword, number):


        PROJECT_ROOT_DIR = "."
        keyword_file = keyword + f"_page_{number}.txt"
        words_path = os.path.join(PROJECT_ROOT_DIR, "data", "BBC_pages", f"{keyword}")

        try:

            with open(os.path.join("data", "BBC_pages",f"{keyword}_pages", f"{keyword_file}"), encoding="utf-8") as f:
                data = soup(f.read(), "html.parser")
                return data
        except:
            print("file open issue")



    def saveCSVFile(self, name, contents, subfolderName=None, baseFolder=None):
        """
        saves a CSV file to the directory specified
        :param name:
        :param contents:
        :param subfolderName:
        :param baseFolder:
        :return:
        """


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
            pd.DataFrame(contents).to_csv(os.path.join(save_path, f"{name}.csv", ))
        except:
            print("file write issue")

    def downloadFile(self, url, CHAPTER_ID=None):
        """
        dowloads the resource at the specified URL to the file directory specified
        :param url:
        :param CHAPTER_ID:
        :return:
        """


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

    @staticmethod
    def keyWordWebpageContent():
        """ PROBLEM 1
        goes though all keywords
        finds the first 100 most relevant bbc articles and downloads pages
        downloads the articles contents and stores in data/Articles
        :result: article content stored in the file system
        """
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
            links = links[:100]
            for link in links:
                print(int((i / len(links)) * 100), "%, of:", wordnum, "/", len(list(keywords)), "keywords")
                i += 1

                content = BBCextraction().getPage(link, None)
                fileMangagement().saveTextFile(f"{word}_page_{i}", str(content), f"{word}_pages", "BBC_pages")

    @staticmethod
    def articleContentCollection():
        """ PROBLEM 2
        goes though all keywords
        finds the first 100 most relevant bbc articles and extracts the article contents
        downloads the articles contents and stores in data/Articles
        :result: article content stored in the file system
        """
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


            PROJECT_ROOT_DIR = "."
            words_path = os.path.join(PROJECT_ROOT_DIR, "data", "BBC_pages", f"{word}_pages")
            link_num = len([name for name in os.listdir(words_path) if os.path.isfile(os.path.join(words_path, name))])

            for link in range(1, link_num+1):
                print(((link ), "%, of:", wordnum, "/", len(list(keywords)), "keywords"))

                content = fileMangagement().openSavedPage(word, link)

                externalLinks, article = BBCextraction().extractArticle(content)
                fileMangagement().saveTextFile(f"{word}_article_{link}", article, f"{word}_articles", "Articles")

                prominent_words.extend(tokenizeString(article))

            fileMangagement().saveCSVFile(f"{word}_prominent_words", list(set(prominent_words)), f"{word}",
                                          "Prominent_words")


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


        url = "http://nlp.stanford.edu/data/glove.6B.zip"

        # checks to see if has downloaded file

        my_file = Path("models/pretrainedModels/glove.6B.zip")
        if not (my_file.is_file()):
            print("File not accessible")
            print("Downloading resource file. This may take roughly 10 minutes depending on connection speeds!")
            fileMangagement().downloadFile(url, "pretrainedModels")

        # if not unzipped, unzips zip file
        my_file = Path("glove.6B.50d.txt")
        if not (my_file.is_file()):
            print("extracting ZIP")

            with zipfile.ZipFile("models/pretrainedModels/love.6B.zip", 'r') as zip_ref:
                zip_ref.extractall(".")

        my_file = Path("glove.6B.50d.txt")

        if not (my_file.is_file()):
            print("files Acquired UNsuccessfully")

    @cache
    def vectoriseString(self, string):
        """
        uses the glove dataset to transform a word into a vector
        :param string: the string to be vecorised
        :return: vecotr as an array of ints
        """
        # checks to see if correct resources have been downloaded, and gets them if nessisary
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

        # now should have required files!

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
            if not (w_vec is None):
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
        return self.avgVecKeywords(word, 50)

    def cosineSimilarity(self, setA_keywords, setB_keywords):

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
        corpus = list(corpus for corpus, _ in groupby(corpus))

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

            distance = similarity(words_dict, article_dict).jaccardSimilarity(set(words_dict[keywordA]["0"]) - common,
                                                                              set(words_dict[keywordB]["0"]) - common)
            if distance == 1.0:
                res[keywordA][word_index] = 1.0
            else:
                res[keywordA][word_index] = distance

    return res


def semanticSimDF(words_dict, article_dict, corpus, keywords, justVecs=False):
    keywordlist = list(keywords)
    res = pd.DataFrame([[0.0 for i in range(len(keywordlist))] for i in range(len(keywordlist))], columns=keywordlist)
    workspace = similarity(words_dict, article_dict)

    if justVecs:
        labels = list(chain.from_iterable([[k_word for article in corpus[k_word]] for k_word in keywordlist]))
        vecs = list(chain.from_iterable(
            [[workspace.avgVecKeywords(article, 50) for article in corpus[k_word]] for k_word in keywordlist]))
        return vecs, labels

    for keywordA in list(keywords):
        for keywordB in list(keywords):
            word_index = keywordlist.index(keywordB)

            distance = workspace.cosineSimilarity(words_dict[keywordA], words_dict[keywordB])

            res[keywordA][word_index] = distance

    return res


def Doc2VecDF(corpus, article_dict, keywords, justVecs=False):
    keywordlist = list(keywords)
    vector_res = pd.DataFrame([[0.0 for i in range(len(keywordlist))] for i in range(len(keywordlist))],
                              columns=keywordlist)

    complexComparisonEngine = Doc2VecSimilarity(corpus, article_dict)

    if justVecs:
        labels = list(chain.from_iterable([[k_word for article in corpus[k_word]] for k_word in keywordlist]))
        vecs = list(chain.from_iterable(
            [[complexComparisonEngine.queryDoc2VecModel(article) for article in corpus[k_word]] for k_word in
             keywordlist]))
        return vecs, labels

    # one vector per keyword
    # vectors = {article : complexComparisonEngine.queryDoc2VecModel(article_dict[article]) for article in keywordlist}
    # one vector per article
    vectors = {k_word: [complexComparisonEngine.queryDoc2VecModel(article) for article in corpus[k_word]] for k_word in
               keywordlist}

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
            newval = (val - min) / (max - min)
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
        try:
            dataframe.index = dataframe.columns
            sns.heatmap(dataframe, vmin=0, vmax=1, annot=True, fmt="f", linewidths=.5)
        except:
            sns.heatmap(dataframe, vmin=0, vmax=1, annot=True, fmt="f", linewidths=.5)

    @staticmethod
    def TwoDRepGraph(vectors, labels, mean=False):


        target = labels

        tsne = TSNE(n_components=2)
        X_train_ = tsne.fit_transform(pd.DataFrame(vectors))

        df_graph = pd.DataFrame(X_train_, columns=['x', 'y'])
        df_graph['keyword'] = labels

        if mean:
            meanpoints = df_graph.groupby('keyword', as_index=False)['x'].mean()
            meanpoints['y'] = (df_graph.groupby('keyword', as_index=False)['y'].mean())['y']

            sns.scatterplot(meanpoints['x'], meanpoints['y'], hue=meanpoints['keyword'], palette='colorblind')

        sns.scatterplot(df_graph['x'], df_graph['y'], hue=df_graph['keyword'], palette='colorblind')

    @staticmethod
    def buildingGraphDF(aritcles_corpus, article_dict, keywords, words_dict, option=0):
        """
        :param aritcles_corpus:
        :param article_dict:
        :param keywords:
        :param option: 1 for doc2vec, anythign else for Glove
        :return:
        """
        # for doc2vec
        if option == 1:
            vecs, labels = Doc2VecDF(aritcles_corpus, article_dict, keywords, True)
        # for GloVe
        else:
            vecs, labels = semanticSimDF(words_dict, article_dict, aritcles_corpus, keywords, True)

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

        overall_mean_df.index = overall_mean_df.columns

        for row in list(overall_mean_df.index):
            for coll in list(overall_mean_df.columns):
                if row == col:
                    overall_mean_df[coll][row] = np.nan

        return overall_mean_df

    @staticmethod
    def keywordExample(dfList, colname):
        # order = Jaccard, GloVe, Doc2Vec
        collumbs = list(dfList[0])
        overall_col = pd.DataFrame()

        overall_col["Jaccard"] = dfList[0][colname]
        overall_col["GloVe"] = dfList[1][colname]
        overall_col["Doc2Vec"] = dfList[2][colname]

        #overall_col.index = overall_col.columns

        return overall_col


def symanticSimilarityScoring(jaccard=True, glove=True, DOCtoVEC=True):
    """question3
    goes though keywords
    applies three distance algorithms to the keywords
    1. jaccard coefficient
    2. GloVe dataset downloaded and applied
    3. DOC2VEC model trained and used to find similarity

    NOTE- GloVe data set is roughly 1Gb to download and 2Gb once extracted
        therefore will take a while to download on first run (will persist for subsequent runs of the code)

        -if for some reason errors come from downloading the GloVE dataset set its correlating argument to FALSE when
            calling the function

    NOTE- distance.xlsx should be placed in the same directory as this file
    """

    # gets keywords form the Excell file
    keywords = fileMangagement().openXLS(path)['Keywords']
    aritcles_corpus, article_dict, words_dict = fileMangagement().getContent(keywords)

    list_measuremts = []

    if DOCtoVEC:
        D2V_df = Doc2VecDF(aritcles_corpus, article_dict, keywords)
        D2V_df = normaliseDF(D2V_df)
        list_measuremts.append(D2V_df)

    if jaccard:
        j_df = jacSimDF(words_dict, article_dict, keywords)
        j_df = normaliseDF(j_df)
        list_measuremts.append(j_df)

    if glove:
        s_df = semanticSimDF(words_dict, article_dict, aritcles_corpus, keywords, False)
        s_df = normaliseDF(s_df)
        list_measuremts.append(s_df)

    #overall combination of them all

    overall = visualisation.averageDataFrame(list_measuremts)

    return list_measuremts, overall


def graphing(list_measuremts, plot_individual=False, plot_overall=False, plot_specific=False, specific_name=None, plot_vectorReduction=False, vecReductionOption=1):
    """question 4
    plots the graphs seen in the report
    :param list_measurements: list of measurements dataframes output form question 3
    :param plot_individual: option to plot heatmaps for each measurement algorithum seperately
    :param plot_overall: option to plot heatmap for overall measurement
    :param plot_specific: option to plot the specific measurements for each algoithum for one keyword
    :param specific_name: if plot_specific, name of keyword desired
    :param plot_vectorReduction: option to plot the vector reduction scatterplot
    :param vecReductionOption: if plot_vectorReduction,  choice between glove(0) and doc2vec(1) vectors
    """
    keywords = fileMangagement().openXLS(path)['Keywords']
    aritcles_corpus, article_dict, words_dict = fileMangagement().getContent(keywords)

    if plot_individual:
        for m in list_measuremts:
            visualisation.heatMap(m)
            plt.show()

    if plot_overall:
        overall = visualisation.averageDataFrame(list_measuremts)
        visualisation.heatMap(overall)
        plt.show()

    if plot_specific:
        if specific_name == None:
            keyword = visualisation.keywordExample(list_measuremts, 'DoS attack')
            visualisation.heatMap(keyword)
            plt.show()
        else:
            keyword = visualisation.keywordExample(list_measuremts, specific_name)
            visualisation.heatMap(keyword)
            plt.show()

    if plot_vectorReduction:
        visualisation.buildingGraphDF(aritcles_corpus, article_dict, words_dict, keywords, option=vecReductionOption)
        plt.show()


def main(qOne=False, qTwo=False, qThree=False, qFour=False):
    if qOne:
        dataCollection.keyWordWebpageContent()
    if qTwo:
        print("WARNING from ldzc78: please ensure you have run Question(s): 1 previously")
        dataCollection.articleContentCollection()
    if qThree:
        print("WARNING from ldzc78: please ensure you have run Question(s): 1,2 previously")
        list_measuremts, overall = symanticSimilarityScoring(jaccard=True, glove=True, DOCtoVEC=True)
    if qFour:
        print("WARNING from ldzc78: please ensure you have run Question(s): 1,2 previously")
        print("WARNING from ldzc78: running Q3 to get measurements")
        list_measuremts, overall = symanticSimilarityScoring(jaccard=True, glove=True, DOCtoVEC=True)
        graphing(list_measuremts, plot_individual=True, plot_overall=True, plot_specific=True, specific_name=None,
                 plot_vectorReduction=True, vecReductionOption=1)

if __name__ == '__main__':
    main(qOne=True, qTwo=True, qThree=False, qFour=True)




