# %% [markdown]
# 

# %%
# imports
import typing
import os
import math
import sys
import random
import tensorflow as tf
from tensorflow import keras
import numpy as np
from numpy import genfromtxt
from numpy.lib import recfunctions as rfn
import pandas as pd
import sklearn
from sklearn.preprocessing import MinMaxScaler
from IPython.display import display, Math, Latex
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


# %%
# colab config
#from google.colab import drive
#drive.mount('/content/drive')
#os.chdir('/content/drive/My Drive/cs470')

# %% [markdown]
# # Data Preprocessing
# 
# ## Data Loading
# 
# The dataset was downloaded from Kaggle. It contains 32 Columns of 1,000,000 entries. Since it is 216.59 MB, it may take time to load.
# 
# _in this project, "label" will refer to column titles, not the value of the `fraud_bool` column_

# %%
parsed_data = pd.read_csv("Base.csv") # load csv from drive
display(parsed_data[3:10])
classification = parsed_data["fraud_bool"].astype(np.float32)
labels = parsed_data.keys()

# %% [markdown]
# Our dataset is very imbalanced. This reflects the fact that fraudulence is fairly rare in the real world.

# %%
Latex(f'In the complete data set, {classification.sum()} are fraud ({100*classification.sum()/len(classification):.3f}%)')

# %% [markdown]
# # One-hot vectorization
# 
# Machine learning algorithms have a hard time understanding strings. To confront this, we replace a column of multiple unique string values with multiple columns for each unique category. These columns contain a boolean to indicate which category it was. This is alternatively known as a _One-hot_.

# %%
# one-hotify labels
labels_categorical = ["payment_type", "employment_status", "housing_status", "source", "device_os"] # list that contains columns to be vectorized
labels_vectorized = []
for label in labels_categorical:
    label_index = parsed_data.columns.get_loc(label)
    column_vectorized = pd.get_dummies(parsed_data[label])
    for label_vectorized in column_vectorized:
        # new label joins the category with the original column name
        label_vectorized_new = label + "_" + label_vectorized
        parsed_data.insert(
            label_index,
            label_vectorized_new,
            column_vectorized[label_vectorized])
        labels_vectorized.append(label_vectorized_new)
    del parsed_data[label]
# prove that removal occurred and that new columns were added
assert("payment_type" not in parsed_data.columns and "device_os" not in parsed_data.columns and "device_os_windows" in parsed_data.columns)

# %% [markdown]
# # Minimum-Maximum Normalization
# 
# The backpropagation algorithm will eventually attempt to normalize the range of values within a column to become from 0 to 1. We can save time by preprocessing the data beforehand, shaving seconds if not minutes off of training.

# %%
# minimum-maximum normalization
normalized_features = pd.DataFrame(MinMaxScaler().fit_transform(parsed_data),columns=parsed_data.columns)
display(normalized_features[3:10])

# %% [markdown]
# # Data Partitioning
# 
# Finally, the data will be partitioned into both training and evaluation subsets. As suggested in the NeurIPS paper, we use the first 10 months of data for training and the most recent two for evaluation.

# %%
data_training = normalized_features[normalized_features['month'] <= 10./12.]
labels_training = data_training['fraud_bool']

display(data_training[3:10])
data_evaluation = normalized_features[normalized_features['month']  > 10/12]
labels_evaluation  = data_evaluation['fraud_bool']

# these aren't features
_ = [data_evaluation.pop(c) for c in {"month", "fraud_bool"}]
_ = [data_training.pop(c) for c in {"month", "fraud_bool"}]

print(f"Training on {100*(len(data_training)/len(parsed_data)):.2f}% of the data.")

# %% [markdown]
# # Data exploration
# 
# We first separate out the fraudulent and nonfraudulent rows into their own frames:

# %%
fraud = []
fraud_not = []
for rowix,row in normalized_features.iterrows():
    # faught too hard with pandas to use the boolean column as index
    if classification[rowix]:
        fraud.append(row)
    else:
        fraud_not.append(row)
fraud = pd.DataFrame(fraud,columns=normalized_features.columns)
fraud_not = pd.DataFrame(fraud_not,columns=normalized_features.columns)

# %% [markdown]
# Now, we will do some plots of the bivariate and univariate distributions of some pairs of columns. Ember thinks this is useful for exploration, but it takes a few minutes to run.

# %%
labelscopy = list(normalized_features.columns)
random.shuffle(labelscopy)
# plots two features' distributions on a single plot
def distribution(dataset, labels, label_0, label_1):
    args = { 'kind':'hist', 'xlim':(0,1), 'ylim':(0,1) }
    g1 = sns.jointplot(x=fraud.loc[:,label_0], y=fraud.loc[:,label_1], color = 'red', **args)
    plt.suptitle("Fraudulent distribution")
    g2 = sns.jointplot(x=fraud_not.loc[:,label_0], y=fraud_not.loc[:,label_1], color='blue',
                **args)
    plt.suptitle("Non-fraudulent distribution")

try:
    while len(labelscopy) != 0:
        label_0 = labelscopy.pop()
        label_1 = labelscopy.pop()
        distribution(normalized_features, classification, label_0, label_1)
except IndexError:
    pass

# %%
plt.close()

# %% [markdown]
# # Model Design
# 
# We will be using a simple ANN for this project. The data is categorical (remapped to one-hots) and numerical (now min-max normalized), with no spatial significance. Ember wanted to experiment with neural decision forests, but there was no interest.

# %%
# model design

model = keras.Sequential([
    keras.layers.Dense(512, activation="relu", input_shape=(data_training.shape[-1],)),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(256, activation="relu"),
    keras.layers.Dropout(0.1),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])

# %%
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.FalsePositives(), tf.keras.metrics.FalseNegatives(), tf.keras.metrics.TrueNegatives(), tf.keras.metrics.TruePositives(), tf.keras.metrics.AUC(curve="PR", num_thresholds=50), tf.keras.metrics.BinaryAccuracy(),tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
model.summary()

# %% [markdown]
# # Training

# %%
weight_for_1 = labels_training.sum() / len(labels_training)
weight_for_0 = 1 - weight_for_1
class_weight = {0.: weight_for_0, 1.: weight_for_1}
display(class_weight)
history = model.fit(
    x=data_training,
    y=labels_training,
    class_weight=class_weight,
    epochs=1)

# %%
model.save("model_0.h5")

# %%
for datapoint in random.choices(data_evaluation, k=5):
    print(f"Model predicts {model.predict([datapoint])[0][0]} for {datapoint}")
for datapoint in random.choices(fraud, k=5):
    print(f"Model predicts {model.predict([datapoint])[0][0]} for fraudulent {datapoint}")
for datapoint in random.choices(fraud_not, k=5):
    print(f"Model predicts {model.predict([datapoint])[0][0]} for non-fraudulent {datapoint}")

history = model.evaluate(data_evaluation, labels_evaluation)
display(history)

# %%



