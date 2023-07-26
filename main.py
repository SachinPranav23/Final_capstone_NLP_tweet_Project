import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

# Step 1: Data Loading and Preprocessing
# Load the dataset into a pandas DataFrame
df = pd.read_csv('/Users/sachinpranav/Downloads/FinalBalancedDataset.csv')
print(df.columns)
# Perform data preprocessing if needed
# Check for missing values
print(df.isnull().sum())

# Drop any rows with missing values
df.dropna(inplace=True)

# Step 2: Text Cleaning and Vectorization
# Define a function to clean the text data (optional)
def clean_text(tweet):

		# Convert to lowercase
		tweet = tweet.lower()
		# Remove special characters and numbers
		tweet = re.sub(r'[^\w\s]', '', tweet)
		# Remove digits
		tweet = re.sub(r'\d', '', tweet)
		# Remove extra whitespaces
		tweet = re.sub(r'\s+', ' ', tweet).strip()
		# Remove punctuation
		tweet = tweet.translate(str.maketrans('', '', string.punctuation))
		# Tokenize text
		tokens = word_tokenize(tweet)
		# Remove stopwords
		stop_words = set(stopwords.words('english'))
		filtered_tokens = [token for token in tokens if token not in stop_words]
		# Join tokens back to form the cleaned text
		cleaned_tweet = ' '.join(filtered_tokens)
		return cleaned_tweet

# Apply the text cleaning function to the 'text' column
df['cleaned_tweet'] = df['tweet'].apply(clean_text)


# Step 2: Text Vectorization
# Choose either Bag of Words (BoW) or TF-IDF
vectorizer = CountVectorizer()  # or TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_tweet'])
y = df['Toxicity']
#
# # Step 3: Model Building and Evaluation
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and evaluate models on the training set
train_models = [
    DecisionTreeClassifier(),
    RandomForestClassifier(),
    MultinomialNB(),
    KNeighborsClassifier(),
    SVC()
]

print("Training Results:")
for model in train_models:
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)

    precision_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred)
    conf_matrix_train = confusion_matrix(y_train, y_train_pred)
    roc_auc_train = roc_auc_score(y_train, y_train_pred)

    print(f"Model: {model.__class__.__name__}")
    print(f"Precision (Train): {precision_train}")
    print(f"Recall (Train): {recall_train}")
    print(f"F1-Score (Train): {f1_train}")
    print(f"Confusion Matrix (Train):\n{conf_matrix_train}")
    print(f"ROC-AUC Score (Train): {roc_auc_train}")
    print("\n")

# Evaluate models on the testing set
print("Testing Results:")
for model in train_models:
    y_test_pred = model.predict(X_test)

    precision_test = precision_score(y_test, y_test_pred)
    recall_test = recall_score(y_test, y_test_pred)
    f1_test = f1_score(y_test, y_test_pred)
    conf_matrix_test = confusion_matrix(y_test, y_test_pred)
    roc_auc_test = roc_auc_score(y_test, y_test_pred)

    print(f"Model: {model.__class__.__name__}")
    print(f"Precision (Test): {precision_test}")
    print(f"Recall (Test): {recall_test}")
    print(f"F1-Score (Test): {f1_test}")
    print(f"Confusion Matrix (Test):\n{conf_matrix_test}")
    print(f"ROC-AUC Score (Test): {roc_auc_test}")
    print("\n")

#
#     # Plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_test_pred)
    plt.plot(fpr, tpr, label=model.__class__.__name__)

plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
