# %%
import pandas as pd 
import matplotlib.pyplot as plt 

from scipy import stats
import numpy as np 

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


import tensorflow as tf 


# %% [markdown]
# # Data Selection
#Import Dataset

# %%
charity_df = pd.read_csv('./Resource/charity_data.csv')
charity_df.head()

# %%
charity_df.isnull().sum()    # no null values in entire dataframe
charity_df.dtypes

# %%
# remove inactivity companies (status is 0)
charity_df = charity_df[charity_df.STATUS == 1]
# %%
# store the names of companies in a new dataframe
companies_name_df = charity_df[['EIN','NAME']]
companies_name_df.head()

# %%
# remove some irrelevant variables, neither features nor taget for model
charity_df = charity_df.drop(columns = ['EIN','NAME', 'STATUS','SPECIAL_CONSIDERATIONS'])
charity_df.head(3)


# %% [markdown]
# # Data Preprocess PART 1
# Transform categorical variables into numerical values
# %%
# Generate a name list of categorical variables 
cat_name_list = charity_df.dtypes[charity_df.dtypes == 'object'].index.tolist()
# check the unique values in each columns for bucketing (APPLICATION_TYPE has 17, CLASSIFICATION: 71)
charity_df[cat_name_list].nunique()

# %%
# check unique values detail and density for APPLICATION_TYPE column
APPL_column_Series = charity_df['APPLICATION_TYPE'].value_counts()
APPL_column_Series
# %%
APPL_column_Series.plot.density()

# %%
# The cutoff point should be around 200 counts, btw T10 and T9, bucketing them in 'others' 
# Combine rare categorical values via bucketing.
replace_APPL_list = APPL_column_Series[APPL_column_Series < 200].index.tolist()

for app_type in replace_APPL_list:
    charity_df['APPLICATION_TYPE'] = charity_df['APPLICATION_TYPE'].replace(app_type, 'Others')

charity_df['APPLICATION_TYPE'].value_counts()
# %%
# check unique values detail and density for CLASSIFICATION column
CLASS_column_Series = charity_df['CLASSIFICATION'].value_counts()
CLASS_column_Series
# %%
CLASS_column_Series.plot.density()
# %%
# The cutoff point should be around 200 counts, btw C4000 and C1700, bucketing them in 'others' 
replace_CLASS_list = CLASS_column_Series[CLASS_column_Series < 200].index.tolist()

for class_type in replace_CLASS_list:
     charity_df['CLASSIFICATION'] = charity_df['CLASSIFICATION'].replace(class_type, 'Others')

charity_df['CLASSIFICATION'].value_counts()

# %%
# double check every categorical variables doesn't exceed 10 unique values
charity_df[cat_name_list].nunique()

# %%
# use oneHot encode method to convert categorcal variables into several numerical variables
enc = OneHotEncoder(sparse = False) # return array not sparse matrix

# create a encode new DataFrame only contains encoded categorical variables
encode_df = pd.DataFrame(enc.fit_transform(charity_df[cat_name_list]))

encode_df.columns = enc.get_feature_names(input_features = cat_name_list)

encode_df.head()

# %%
# merge back to orginal df and drop original unencoded columns
encoded_charity_df  = charity_df.merge(encode_df, left_index = True, right_index = True)\
                                .drop(columns = cat_name_list)
encoded_charity_df.head()

# %% [markdown]
# # Data Selection
# Redece outliers and noisy data points for specific variables
# %%
# focus on numerical variable: ASK_AMT 
encoded_charity_df.ASK_AMT.describe()

# %%
encoded_charity_df.ASK_AMT.plot.box()
# %%
# According to boxplot, IQR medthod for outliers has huge outliers(8205), it's not good for this case

Q1 = encoded_charity_df.ASK_AMT.quantile(0.25)
Q3 = encoded_charity_df.ASK_AMT.quantile(0.75)
IQR = Q3 - Q1

boo_ASK = (encoded_charity_df.ASK_AMT < (Q1 - 1.5 * IQR)) |(encoded_charity_df.ASK_AMT > (Q3 + 1.5 * IQR))
IQR_outliers = encoded_charity_df.ASK_AMT[boo_ASK == True]
len(IQR_outliers)

# %%
# perform z-score to filter outliers

ASK_AMT_Zscore = np.abs(stats.zscore(charity_df.ASK_AMT) <= 3)

outliers = ASK_AMT_Zscore[ASK_AMT_Zscore == False]

len(outliers)
# %%
# remove 53 outliers from dataframe
encoded_charity_df = encoded_charity_df[(np.abs(stats.zscore(encoded_charity_df.ASK_AMT)) <= 3)]
encoded_charity_df.head()

# %%
encoded_charity_df.describe()
# %% [markdown]
# # Data Preprocess PART 2
# Determine features (X) and target column (y) and split into training, testing data
# %%
y = encoded_charity_df['IS_SUCCESSFUL'].values

X = encoded_charity_df.drop(columns = ['IS_SUCCESSFUL']).values

# split into training and testing parts
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 42)

# %% [markdown]
# # Data Preprocess PART 3
# Standardize all features variable before enter into model
# %%
scaler = StandardScaler()
scaler.fit(X_train)

X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)

# %% [markdown]
# # (Removed) Data Preprocess PART 4 PCA:  no improvement for optimazation

# %%
#from sklearn.decomposition import PCA

#pca = PCA(n_components = 3, random_state= 42)
# train pca model with scaled data
#pca.fit(X_train_scaled)
# X_pca_train = pca.transform(X_train_scaled)
#X_pca_test =pca.transform(X_test_scaled)
#print(f'The pca ratio is {pca.explained_variance_ratio_}')


# %% [markdown]
# # Deep_Learning Neural Network
# Use TensorFlow neural network design a binary classification model that can predict if 
# a previously funded organization will be successful based on the features in the dataset.


# %%
# determine number of neurons in each layers
num_input = len(X_train_scaled[0])
num_first = len(X_train_scaled[0])*2
num_second = 50
num_third = 20

#kernel_reg = tf.keras.regularizers.l2(0.01)
#act_reg = tf.keras.regularizers.l1(0.01)

# build a Sequential model as a base
nn_model = tf.keras.models.Sequential()

# build Dense layer for input and first hidden layer
nn_model.add(tf.keras.layers.Dense(units=num_first, input_dim = num_input,
                                    activation ='relu'))
                                    #kernel_regularizer = kernel_reg,activity_regularizer= act_reg))
# seconde hidden layer
nn_model.add(tf.keras.layers.Dense(units = num_second, activation='relu'))
# THIRD hidden layer
nn_model.add(tf.keras.layers.Dense(units = num_third, activation='relu'))
# now add a ReLU layer explicitly:
#nn_model.add(tf.keras.layers.LeakyReLU(alpha=0.05))
# output layer
nn_model.add(tf.keras.layers.Dense(units = 1, activation='sigmoid'))

nn_model.summary()
# %%
# config setting
nn_model.compile(loss = 'binary_crossentropy', optimizer = 'adam',
                metrics = ['accuracy'])
# %%
# train model with training data
model_history = nn_model.fit(X_train_scaled, y_train, epochs=200)

history_df = pd.DataFrame(model_history.history, 
                    index = range(1, len(model_history.history['loss'])+1))

# viz the loss 
history_df.plot(y = 'accuracy')
# %%
# evaluate with test data
model_loss, model_accuracy = nn_model.evaluate(X_test_scaled, y_test,
                                                verbose = 2)
print(f"The model's Loss is {model_loss}, Accuracy is {model_accuracy}")


# %%
