import pandas as pd
import numpy as np
import pickle

kyphosis_df = pd.read_csv('kyphosis.csv')

from sklearn.preprocessing import LabelEncoder

LabelEncoder_y = LabelEncoder()
kyphosis_df['Kyphosis'] = LabelEncoder_y.fit_transform(kyphosis_df['Kyphosis'])

X = kyphosis_df.drop(['Kyphosis'], axis = 1)
y = kyphosis_df['Kyphosis']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)


from sklearn.tree import DecisionTreeClassifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, y_train)

pickle.dump(decision_tree , open('kyphosis.pkl', 'wb'))