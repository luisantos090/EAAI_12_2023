# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 21:59:30 2023

@author: pinhosl3
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
# import tensorflow as tf 
# # Load TensorFlow Decision Forests
# import tensorflow_decision_forests as tfdf
# import the regressor 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.utils import shuffle
from sklearn.tree import export_graphviz  
from sklearn.model_selection import train_test_split
from sklearn import tree


path = r"C:\Users\pinhosl3\OneDrive - London South Bank University\Desktop\Decision Trees"
os.chdir(path)

def plot_predictions(test_labels, test_predictions, lims):
    plt.scatter(test_labels, test_predictions)
    plt.xlim(lims)
    plt.ylim(lims)
    # plt.axes(aspect='equal')
    plt.xlabel('Measured Nu [kN]')
    plt.ylabel('Predicted Nu [kN]')
    
    plt.plot(lims, lims)
    # plt.plot([1.5*i for i in lims], lims)
    # plt.plot([0.5*i for i in lims], lims)
    # plt.plot([2.0*i for i in lims], lims)
    # plt.plot([0.1*i for i in lims], lims)
    plt.figtext(0.5,0.7,"R² = 0.9988" , fontsize=12)
    plt.show()

filename = "data.xlsx"

df = pd.read_excel(path+"/"+filename)
df = df[df.columns].dropna(axis=1)

# # create a regressor object 
regressor = DecisionTreeRegressor()  

target = df.pop('Max_Load (FEA)').values
features = df.values

features, target = shuffle(features, target)
  


x_train, x_test, y_train, y_test = train_test_split(features, target, test_size=0.5)


# # fit the regressor with X and Y data 
fit = regressor.fit(x_train, y_train) 

# # test the output by changing values, like 3750 
y_pred = regressor.predict(x_test) 

print(regressor.score(x_train, y_train))
print(regressor.score(x_test, y_test))
print(regressor.get_depth())
print(regressor.get_n_leaves())

pd.DataFrame(x_test).to_excel('x_test.xlsx', index=False)
pd.DataFrame(x_train).to_excel('x_train.xlsx', index=False)
pd.DataFrame(y_test).to_excel('y_test.xlsx', index=False)
pd.DataFrame(y_train).to_excel('y_train.xlsx', index=False)
pd.DataFrame(y_pred).to_excel('y_pred.xlsx', index=False)



fig = plt.figure(figsize=(25,20))
_ = tree.plot_tree(regressor,
                    feature_names=list(df.columns))
                    # class_names=iris.target_names,
                    # filled=True)
fig.savefig("decistion_tree.png")

filename1 = "y_pred.xlsx"
filename2 = "y_test.xlsx"

y_pred = pd.read_excel(path+"/"+filename1).values
y_test = pd.read_excel(path+"/"+filename2).values
                       


lims = [min(np.min(y_test), np.min(y_pred)), max(np.max(y_test), np.max(y_pred))]
plot_predictions(y_test, y_pred, lims)
