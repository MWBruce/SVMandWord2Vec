####################################################################################
##
##  Disc: My combination of word2vec and svm to develop a machine learning model 
##  that predicts stock price based on reddit posts on any given day
##
##  Credit to the StatQuest YouTube Channel for lots of SVM code
##
##
####################################################################################
import gensim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA

# Load RedditData.csv and train a Word2Vec model
df = pd.read_csv("RedditData.csv", header=0, sep=',')

df_1 = resample(df[df['Price_Change'] == 1], n_samples=5000, replace=True, random_state=42)
df_0 = resample(df[df['Price_Change'] == 0], n_samples=5000, replace=True, random_state=42)
df_minus1 = resample(df[df['Price_Change'] == -1], n_samples=5000, replace=True, random_state=42)

df = pd.concat([df_1, df_0, df_minus1])

review_text = df['text'].apply(gensim.utils.simple_preprocess)
model = gensim.models.Word2Vec(
    window=10,
    min_count=2,
    workers=4
)
model.build_vocab(review_text, progress_per=1000)
model.train(review_text, total_examples=model.corpus_count, epochs=model.epochs)
model.save("./Save.model")
# print(model.wv.most_similar("profit"))

# Convert each document to a vector using the trained Word2Vec model
doc_vectors = []
for doc in review_text:
    doc_vec = np.zeros(100)
    word_count = 0
    for word in doc:
        if word in model.wv.key_to_index:
            doc_vec += model.wv[word]
            word_count += 1
    if word_count > 0:
        doc_vec /= word_count
    doc_vectors.append(doc_vec)

# Use the document vectors to train an SVM
X = np.array(doc_vectors)
y = df['Price_Change'].copy()

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
X_train_scaled = scale(X_train)
X_test_scaled = scale(X_test)

clf_svm = SVC(random_state=42)
clf_svm.fit(X_train_scaled, y_train)

# Plot the confusion matrix
y_pred = clf_svm.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=["Down", "Same", "Up"], yticklabels=["Down", "Same", "Up"],
       title="Confusion Matrix",
       ylabel="True label",
       xlabel="Predicted label")

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
plt.savefig('confusion_matrix.png')
# plt.show()

phrase = "the price will go down"
phrase_vec = np.zeros(100)
word_count = 0
for word in gensim.utils.simple_preprocess(phrase):
    if word in model.wv.key_to_index:
        phrase_vec += model.wv[word]
        word_count += 1
if word_count > 0:
    phrase_vec /= word_count

scaled_phrase_vec = scale(phrase_vec.reshape(1, -1))
prediction = clf_svm.predict(scaled_phrase_vec)

print("Prediction for the phrase '{}': {}".format(phrase, prediction[0]))

param_grid = [
    {'C': [0.5,1,10,100],
     'gamma': ['scale',1,0.1,0.001,0.0001],
     'kernel': ['rbf']},
]

optimal_params = GridSearchCV(
    SVC(),
    param_grid,
    scoring = 'accuracy', #Other scoring ones include 'balanced_accuracy','f1', 'f1_micro', 'f1_macro', 'fa_weighted', 'roc_auc'
    verbose=0
)

optimal_params.fit(X_train_scaled,y_train)
print(optimal_params.best_params_) #will tell us ideal value for C

clf_svm = SVC(random_state=42, C=100, gamma=0.001)
clf_svm.fit(X_train_scaled, y_train)

# Plot the confusion matrix
y_pred = clf_svm.predict(X_test_scaled)
cm = confusion_matrix(y_test, y_pred)

fig, ax = plt.subplots()
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=["Down", "Same", "Up"], yticklabels=["Down", "Same", "Up"],
       title="Confusion Matrix",
       ylabel="True label",
       xlabel="Predicted label")

thresh = cm.max() / 2.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

fig.tight_layout()
plt.savefig('confusion_matrix2.png')