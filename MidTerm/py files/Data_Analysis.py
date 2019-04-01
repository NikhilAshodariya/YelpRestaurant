import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
import string
import collections

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
from nltk import sent_tokenize
from nltk import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
stop_words = stopwords.words('english')
stop_words += ["restaurant", "restaurants", "food", "would", "u", "n't", "ve"]
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


def analysis(file_Path):
    reviews_data = pd.read_csv(file_Path, header=0)
    reviews_data.head()

    # Summary of the Reviews dataset
    reviews_data.info()

    reviews_data.describe()

    # Corerraltion between numerical data (cool, funny, useful votes and star rating)
    reviews_data.corr()

    # Cool, funny and useful describes the number of cool, funny and useful votes given to the review.
    # From the above dataset description and correlation, it is observed that less than 25% of the reviewers had voted the reviews as cool, funny or useful.
    # So, the cool, funny and useful votes data is not useful here.

    # Dropping cool, date, funny and useful columns
    reviews_data.drop(["cool", "date", "funny", "useful"], axis=1, inplace=True)
    reviews_data.sample()

    # ## Data Visualization

    # counts the star rating in the reviews dataset
    reviews_data.stars.value_counts()

    # plotting the star rating againt it's count
    ax = reviews_data.stars.value_counts().plot.bar(
        figsize=(6, 4), title="Model count by stars for Reviews")
    ax.set(xlabel="stars", ylabel="model count")

    # From the above chart, we can analyze that the majority of reviews have received 5 start ratings...
    #
    # ### Generating a word cloud for more visualization
    # -  Word cloud without lemmatiztion to analyze the top words used in the reviews.

    # Reviews without lemmatization
    text = " ".join(x for x in reviews_data.text)

    # Create stopword list:
    stop_words = set(STOPWORDS)
    stop_words.update(["restaurant", "restaurants", "food"])

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stop_words, min_font_size=10,
                          background_color="white").generate(text)

    # Display the generated image:
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # Save the image in the img folder:
    # wordcloud.to_file("Images/before_lemmatization.png")

    # Loading the Business dataset for restaurants in Illinois
    business_data = pd.read_csv(
        "C:\\Users\\Nikhil\\Documents\\Python Notebooks\\BIA_660_B(Web_Mining_Notebook)\\project\\submit\\data\\IL_BusinessData.csv", header=0)

    # Snapshot of Business dataset
    business_data.head()

    # Dropping attributes column
    business_data.drop(["attributes"], axis=1, inplace=True)

    # Sample business dataset
    business_data.sample()

    # Summary of the business dataset
    business_data.info()

    business_data.describe()

    # counts the star rating in the business data
    business_data.stars.value_counts()

    # correlation between numeric data (Review_count and star rating) in the business dataset
    business_data.corr()

    # plotting the star rating againt it's count for business data
    ax = business_data.stars.value_counts().plot.bar(
        figsize=(6, 4), title="Model count by stars for Businesses")
    ax.set(xlabel="stars", ylabel="model count")
    plt.show()

    analysisHelper(file_Path)


def analysisHelper(file_Path):
    df = pd.read_csv(file_Path)

    df.head(2)
    df.drop(["cool", "date", "funny", "useful"], axis=1, inplace=True)
    df.sample()
    df[["business_id", "text"]].as_matrix()[0]

    processedData = list(df[["business_id", "text"]].as_matrix())

    # ## Tokenization
    WNlemma = nltk.WordNetLemmatizer()

    def tokenize(x):
        res = []
        ans = []
        ans.append(x[0])
        data = x[1]

        for val in sent_tokenize(data):
            val = val.strip(string.punctuation).lower()
            filtered_text = [w for w in word_tokenize(val) if not w in stop_words]

            lemmatized_tokens = [WNlemma.lemmatize(
                w.strip(string.punctuation)) for w in filtered_text if w.strip(string.punctuation) != ""]

            res = res+lemmatized_tokens

        ans.append(res)

        return ans

    clean_reviews = list(map(tokenize, processedData))
    clean_text = []
    for x in clean_reviews:
        clean_text.append(x[1])
    # print (clean_text)
    token_list = []
    for x in clean_text:
        token_list = token_list + x
    # print(token_list)
    # print (stop_words)

    # Generating a word cloud for more visualization
    # - Word cloud with stop words and with lemmatiztion to analyze the top words used in the reviews.

    text = ""
    for x in token_list:
        text = text + " " + x

    # Generate a word cloud image
    wordcloud = WordCloud(stopwords=stop_words, min_font_size=10,
                          background_color="white").generate(text)

    # Display the generated image:
    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()

    # Save the image in the img folder:
    # wordcloud.to_file("Images/after_lemmatization.png")

    # Plotting top 20 words used in the reviews
    cv = CountVectorizer()
    bow = cv.fit_transform(token_list)
    # print (cv.get_feature_names())
    word_freq = dict(zip(cv.get_feature_names(), np.asarray(bow.sum(axis=0)).ravel()))
    word_counter = collections.Counter(word_freq)
    word_counter_df = pd.DataFrame(word_counter.most_common(20), columns=['word', 'freq'])
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.barplot(x="word", y="freq", data=word_counter_df, palette="PuBuGn_d", ax=ax)
    plt.show()


if __name__ == '__main__':
    print("program started")
    FILE_PATH = "C:\\Users\\Nikhil\\Documents\\Python Notebooks\\BIA_660_B(Web_Mining_Notebook)\\project\\submit\\data\\il_reviews.csv"
    analysis(FILE_PATH)
