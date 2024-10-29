import pandas as pd
from gensim import corpora
from gensim import models
from pprint import pprint  # pretty-printer
from gensim import similarities
import re
from nltk.corpus import stopwords
from nltk import PorterStemmer
import datetime
import numpy as np

def get_all_values_in_column(df: pd.DataFrame, target_column: str):
    # Split the tags column by comma, expand it into a list, and flatten the resulting lists
    tags_series = df[target_column].dropna().str.split(',')
    all_tags = [tag.strip() for sublist in tags_series for tag in sublist]

    # Remove duplicates by converting the list to a set
    return list(set(all_tags))

def get_articles_by_topics(df: pd.DataFrame, target_topics: list[str]):
    pattern = '|'.join(target_topics)
    return df[df['article_section'].str.contains(pattern, case=False, na=False)]
 
def is_article_about_topics(df: pd.Series, target_topics: list[str]):
    return pd.notna(df['article_section']) and any(topic.lower() in df['article_section'].lower() for topic in target_topics)

# Creating the model (from the program begin to the call similarities.MatrixSimilarity(tfidf_vectors))
def create_model(df: pd.DataFrame, main_column, porter, stoplist, isLDA: bool):    
    # Extract documents from df
    documents = df[main_column].dropna().to_list()
    print(f"Vectorizing {len(documents)} documents based on '{main_column}'...")

    # remove common words and tokenize
    texts = [
        [porter.stem(word) for word in document.lower().split() if word not in stoplist]
        for document in documents
    ]

    # print("Tokens of each document:")
    # pprint(texts)
    # input(".....")

    # create mapping keyword-id
    dictionary = corpora.Dictionary(texts)

    # create the vector for each doc
    model_bow = [dictionary.doc2bow(text) for text in texts]

    # parameters to return
    # vectors = None
    model = None

    if isLDA:
        # create the LDA model from bow vectors
        lda = models.LdaModel(model_bow, num_topics=30, id2word=dictionary, random_state=1, passes=2)
        vectors = []
        for v in model_bow:
            vectors.append(lda[v])

        # print()
        # print("LDA vectors for docs (in terms of topics):")
        # i = 0
        # for v in vectors:
        #     print(v, documents[i])
        #     i += 1

        model = lda
    else:
        # create tfidf model
        tfidf = models.TfidfModel(model_bow)
        vectors = tfidf[model_bow]
        id2token = dict(dictionary.items())
        model = tfidf
        
    matrix = similarities.MatrixSimilarity(vectors)
    print()
    print("Matrix similarities")
    print(matrix)
    print("======================================================\n")
    return vectors, matrix

# Implementation of the pseudocode above.
def calculate_ratio_quality(df: pd.DataFrame, target_topics, vectors, matrix, silent=True, topn=10):
    # Implement the following pseudocode to calculate the variable ratio_quality using the TFIDF vectors:
    total_goods = 0
    filtered_df = get_articles_by_topics(df, target_topics)
    print(filtered_df[['headline', 'article_section']])
    print("======================================================\n")

    start_index = 0

    # For every article (a) on topic "Food and Drink":
    for index, row in filtered_df.iterrows():
        # Obtain the top-10 most similar articles (top-10) in Corpus to a
        vec = vectors[index]
        
        # Calculate similarities between a and each doc of texts using tfidf/lda vectors and cosine
        sims = matrix[vec]

        # sort similarities in descending order
        sims = sorted(enumerate(sims), key=lambda item: -item[1])[start_index:start_index+topn]
        
        if not silent:
            print()
            print("Given the doc: " + row['headline'] + '\n')
            print("The Similarities between this doc and the documents of the corpus are:")
        for doc_position, doc_score in sims:
            similar_article = df.iloc[doc_position]
            
            if not silent:
                print(f"{doc_score} - {similar_article['article_section']} - {similar_article['headline']}")

            # if doc_score == 1:
            #     doc_s = [porter.stem(word) for word in similar_article[main_column].lower().split() if word not in stoplist]
            #     print(doc_s)
            #   input("Continue...")

            # Count how many articles in top-10 are related to topic "Food and Drink" (goods)
            if (is_article_about_topics(similar_article, target_topics)):
                if not silent:
                    print("-------SIMILAR------------")
                total_goods = total_goods+1

    ratio_quality = total_goods/(len(filtered_df)*10)
    return ratio_quality

# Proposal
def new_create_model(df: pd.DataFrame, column_a, column_b, porter, stoplist, isLDA: bool):
    vectors_a, matrix_a = create_model(df, column_a, porter, stoplist, isLDA)
    vectors_b, matrix_b = create_model(df, column_b, porter, stoplist, False) # don't use lda for computing tags-based similarity

    return vectors_a, vectors_b, matrix_a, matrix_b

def new_calculate_ratio_quality(df: pd.DataFrame, target_topics, vectors_a, vectors_b, matrix_a, matrix_b, silent=True, topn=10):
    # Implement the following pseudocode to calculate the variable ratio_quality using the TFIDF vectors:
    total_goods = 0
    filtered_df = get_articles_by_topics(df, target_topics)
    print(filtered_df[['headline', 'article_section']])
    print("======================================================\n")

    start_index = 0
    alpha = 0.8

    # For every article (a) on topic "Food and Drink":
    for index, row in filtered_df.iterrows():
        # Obtain the top-10 most similar articles (top-10) in Corpus to a
        vec_a = vectors_a[index]
        vec_b = vectors_b[index]
        
        # Calculate similarities between a and each doc of texts using tfidf/lda vectors and cosine
        sims_a = matrix_a[vec_a]
        sims_b = matrix_b[vec_b]

        # Calculate the final combined similarity
        sims = alpha * sims_a + (1 - alpha) * sims_b

        # sort similarities in descending order
        sims = sorted(enumerate(sims), key=lambda item: -item[1])[start_index:start_index+topn]
        
        if not silent:
            print()
            print("Given the doc: " + row['headline'] + '\n')
            print("The Similarities between this doc and the documents of the corpus are:")
        for doc_position, doc_score in sims:
            similar_article = df.iloc[doc_position]
            
            if not silent:
                print(f"{doc_score} - {similar_article['article_section']} - {similar_article['headline']}")

            # if doc_score == 1:
            #     doc_s = [porter.stem(word) for word in similar_article[main_column].lower().split() if word not in stoplist]
            #     print(doc_s)
            #   input("Continue...")

            # Count how many articles in top-10 are related to topic "Food and Drink" (goods)
            if (is_article_about_topics(similar_article, target_topics)):
                if not silent:
                    print("-------SIMILAR------------")
                total_goods = total_goods+1

    ratio_quality = total_goods/(len(filtered_df)*10)
    return ratio_quality

def run(main_column: str, second_column: str, target_topics_1, target_topics_2, isLDA: bool, newMethod: bool, silent=True, nrows=None):
    init_t: datetime = datetime.datetime.now()

    # Read the Excel file
    file_path = 'news1.csv'
    df = pd.read_csv(file_path, delimiter=",", nrows=nrows)       
    
    # Pay attention that some rows may don't have description, so we'll not include those articles there
    df = df.dropna(subset=[main_column]).reset_index(drop=True)
    
    print(df.info())
    # print(df.info())
    
    porter = PorterStemmer()
    stoplist = stopwords.words('english')

    if newMethod:
        vectors_a, vectors_b, matrix_a, matrix_b = new_create_model(df, main_column, second_column, porter, stoplist, isLDA)
    else:
        vectors, matrix = create_model(df, main_column, porter, stoplist, isLDA)
    end_creation_model_t: datetime = datetime.datetime.now()
    ###############################################################################################################################

    if newMethod:
        ratio_quality1 = new_calculate_ratio_quality(df, target_topics_1, vectors_a, vectors_b, matrix_a, matrix_b, silent)
    else:
        ratio_quality1 = calculate_ratio_quality(df, target_topics_1, vectors, matrix, silent)
    end_t1: datetime = datetime.datetime.now()

    ###############################################################################################################################

    if newMethod:
        ratio_quality2 = new_calculate_ratio_quality(df, target_topics_2, vectors_a, vectors_b, matrix_a, matrix_b, silent)
    else:
        ratio_quality2 = calculate_ratio_quality(df, target_topics_2, vectors, matrix, silent)
    end_t2: datetime = datetime.datetime.now()

    # get execution time
    elapsed_time_model_creation: datetime = end_creation_model_t - init_t
    elapsed_time_comparison1: datetime = end_t1 - end_creation_model_t
    elapsed_time_comparison2: datetime = end_t2 - end_creation_model_t

    return ratio_quality1, ratio_quality2, elapsed_time_model_creation, elapsed_time_comparison1, elapsed_time_comparison2

def run_experiment(main_column, second_column, target_topics_1, target_topics_2, newMethod, silent, nrows):
    # TF-IDF run
    tfidf_ratio_quality_1, tfidf_ratio_quality_2, tfidf_elapsed_time_model_creation, tfidf_elapsed_time_comparison_1, tfidf_elapsed_time_comparison_2 = run(main_column, second_column, target_topics_1, target_topics_2, isLDA=False, newMethod=newMethod, silent=silent, nrows=nrows)
    # LDA run
    lda_ratio_quality_1, lda_ratio_quality_2, lda_elapsed_time_model_creation, lda_elapsed_time_comparison_1, lda_elapsed_time_comparison_2 = run(main_column, second_column, target_topics_1, target_topics_2, isLDA=True, newMethod=newMethod, silent=silent, nrows=nrows)

    return [
        tfidf_ratio_quality_1, 
        tfidf_ratio_quality_2,
        tfidf_elapsed_time_model_creation,
        tfidf_elapsed_time_comparison_1,
        tfidf_elapsed_time_comparison_2,
        lda_ratio_quality_1,
        lda_ratio_quality_2,
        lda_elapsed_time_model_creation,
        lda_elapsed_time_comparison_1,
        lda_elapsed_time_comparison_2
    ]

def print_result(result):
    print("====================================================================")
    print("TF-IDF Ratio quality               [F&D]:", result[0])
    print("TF-IDF Ratio quality             [SPORT]:", result[1])
    print('TF-IDF Execution time model             :', result[2], 'seconds')
    print('TF-IDF Execution time comparison   [F&D]:', result[3], 'seconds')
    print('TF-IDF Execution time comparison [SPORT]:', result[4], 'seconds')
    print(f"===================================================================")
    print("LDA Ratio quality               [F&D]:", result[-5])
    print("LDA Ratio quality             [SPORT]:", result[-4])
    print('LDA Execution time model             :', result[-3], 'seconds')
    print('LDA Execution time comparison   [F&D]:', result[-2], 'seconds')
    print('LDA Execution time comparison [SPORT]:', result[-1], 'seconds')
    print("====================================================================")

if __name__ == '__main__':
    main_column = "description" # values are "headline", "description", "tags", "article_section"
    second_column = "tags" # values are "headline", "description", "tags", "article_section"
    
    target_topics_a = ["food & drink", "food and drink"]
    target_topics_b = ["sports"]
    silent = True
    newMethod = True
    nrows = None

    result = run_experiment(main_column, second_column, target_topics_a, target_topics_b, newMethod, silent, nrows)

    print_result(result)