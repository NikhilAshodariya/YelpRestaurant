import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

# Snapshot of Reviews dataset
reviews_data = pd.read_csv("data/il_reviews.csv", header = 0)
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
reviews_data.drop(["cool","date","funny","useful"],axis = 1,inplace=True)
reviews_data.sample()

# ## Data Visualization

# counts the star rating in the reviews dataset
reviews_data.stars.value_counts()

# plotting the star rating againt it's count
ax = reviews_data.stars.value_counts().plot.bar(figsize=(6,4), title="Model count by stars for Reviews");  
ax.set(xlabel = "stars", ylabel = "model count");


# From the above chart, we can analyze that the majority of reviews have received 5 start ratings... 
# 
# ### Generating a word cloud for more visualization
# -  Word cloud without lemmatiztion to analyze the top words used in the reviews.

# Reviews without lemmatization
text = " ".join(x for x in reviews_data.text)

# Create stopword list:
stop_words = set(STOPWORDS)
stop_words.update(["restaurant","restaurants", "food"])

# Generate a word cloud image
wordcloud = WordCloud(stopwords = stop_words, min_font_size = 10, background_color = "white").generate(text)

# Display the generated image:
plt.figure(figsize = (8, 8), facecolor = None) 
plt.imshow(wordcloud, interpolation = 'bilinear')
plt.axis("off")
plt.show()

# Save the image in the img folder:
# wordcloud.to_file("Images/before_lemmatization.png")

# Loading the Business dataset for restaurants in Illinois
business_data = pd.read_csv("data/IL_BusinessData.csv", header = 0)

# Snapshot of Business dataset
business_data.head()

# Dropping attributes column
business_data.drop(["attributes"], axis = 1, inplace = True) 

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
ax = business_data.stars.value_counts().plot.bar(figsize=(6,4), title="Model count by stars for Businesses");
ax.set(xlabel = "stars", ylabel = "model count");
plt.show()
