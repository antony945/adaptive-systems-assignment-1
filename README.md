This assignment can be completed in teams of two or three students.


Given this [data set](https://drive.google.com/file/d/1FdqH0yR6ZpIURGO1l5jEnsJZmlj4kHvS/view?usp=sharing) including news, the assignment consists of the following tasks:

1. Implement the following pseudocode to calculate the variable ratio_quality using the TFIDF vectors:
    ```
    total_goods = 0

    # For every article (a) on topic "Food and Drink":

    # Obtain the top-10 most similar articles (top-10) in Corpus to a

    # Count how many articles in top-10 are related to topic "Food and Drink" (goods)

    total_goods = total_goods + goods

    ratio_quality = total_goods/(num_articles_food_and_drink*10)
    ```

    And measure the execution times separately for the following two subprocesses: 
    - Creating the model (from the program begin to the call similarities.MatrixSimilarity(tfidf_vectors))
    - Implementation of the pseudocode above.


2. Repeat the previous task with LDA vectors and compare the results. Explain the differences in the results. Use 30 topics, two passes, and a random state parameter.

3. Repeat the previous two tasks but with the topic "Sports" and compare the results. Why do you think that the quality of the results is worse than the ones obtained with the topic "Food and Drink"?

4. Explain how you can get better results comparing articles by resorting to tagging. Propose a method to calculate the similarity between two articles using their associated tags. Note that the articles in the data set are tagged.

To do this assignment, you must use the python libraries Gensim and NLTK, which provide implementations for the CB algorithms addressed in the course.

You will have to submit a zip/rar file containing a document explaining the results obtained in the proposed tasks, the discussion of these results, and the source code. 

Only one member of each team must submit the assignment in Moodle.


ARTICLE1, FOOD

DESC_WEIGHT = 0.6
TAG_WEIGHT = 0.4

# Combined Similarity Calculation
S_combined = (weight_description * S_description) + (weight_tags * S_tags)

1  0.8 SPORT =>  S(SPORT|FOOD) = 0.2 => 0.8*0.7 + 0.3*0.2 = 0.56 + 0.06 = 0.62
2  0.5 FOOD => S(FOOD|FOOD) = 1 => 0.35+0.3 = 0.888!!!!
