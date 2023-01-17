# importing required packages
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score

def df_info(df):
    """This function takes data frame as input and returns
       structure of the data frame such as columns,head,tail
       ,transpose,summary"""
    print('Columns of the Data Frame\n')
    print(df.columns)
    print('\n\n')
    print('The top values of Data Frame\n')
    print(df.head())
    print('\n\n')
    print('The bottom values of Data Frame\n')
    print(df.tail())
    print('\n\n')
    print(f'The size of the data frame : {df.size}\n')
    print(f'The shape of the data frame : {df.shape}\n')
    print('The transpose of Data Frame\n')
    print(df.T)
    print('\n\n')
    print('summary of the Data Frame\n')
    print(df.info(verbose = True))
    print((df.describe()).T)

def box_plot(df,image_name):
    """This function will take data frame as input
    melts the data according to the class attribute
    and plots the boxplot using seaborn"""
    df_melted = pd.melt(df,id_vars='class',
                          var_name='Attributes',
                          value_name='Value')

    plt.figure()
    sns.boxplot(x='Attributes',y='Value',hue='class',data=df_melted)
    plt.xticks(rotation=75) # rotating xticks by 75 degree
    plt.title('box plot before Standardization') 
    plt.savefig('box plot before Standardization')
    plt.show()

cwd = os.getcwd()
df = pd.read_csv('magic04.data', sep =',') # reading the dataset by using pandas
df_info(df)

# plots the bar graphs according to the count of datasets divided by class attribute
plt.figure()
plt.bar(x='g', height= len(df[df['class']=='g']), label = 'Class g')
plt.bar(x='h', height= len(df[df['class']=='h']), label = 'class h')
plt.xlabel('class')
plt.ylabel('count')
plt.title('Count g & h')
plt.legend()
plt.savefig('count')
plt.show()

# plotting correlation cluster map using seaborn
df_corr = df.corr()
plt.figure()
sns.clustermap(df_corr, annot = True, figsize=(16,10)) # defining figure size
plt.title('Correlation between Attributes', fontdict={'fontsize':'14'})
plt.savefig('Correlation between Attributes')
plt.show()

# plotting box plot before standardization
box_plot(df = df, image_name = 'box plot before Standardization')

x = df.drop(['class'], axis = 1)
y = df['class']
col = x.columns.tolist() # converting columns to list
otl = LocalOutlierFactor() # using Localoutliner to identify outliner values
pred = otl.fit_predict(x)

x_score = otl.negative_outlier_factor_ # scoring the each value
score = pd.DataFrame()
score['score'] = x_score # storing in a dataset

threshold = -1.5 # defining an thrusthold value.
# storing the indexes whose score is less than threshold value.
outlier_index = score[score['score']<threshold].index.tolist() 
# droping the indexes
x = x.drop(outlier_index)
y = y.drop(outlier_index).values
# splitting the data using sklearn
x_train, x_test,  y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.25
                                                    , random_state=42)

# applying the standardization inorder to eliminate the negative values
sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_df =  pd.DataFrame(x_train, columns=col)
x_df['class'] = y_train

# box plot after standardization
box_plot(df = x_df, image_name = 'box plot After Standardization')


def model_NB(model_nb, image_name):
    """In this function we will take model as input and
    fit the data. Performs confusion matrix, prediction 
    on test data and accuracies"""
    model = model_nb.fit(x_train, y_train) # fitting the data
    y_pred = model.predict(x_test)
    y_true = y_test
    cm = confusion_matrix(y_true, y_pred) # forming an confusion matrix using sklearn

    label = ['g','h']
    plt.figure()
    sns.heatmap(cm, annot= True, fmt='.0f', xticklabels = label
                , yticklabels = label)
    plt.title("Confusion matrix for "+image_name)
    plt.savefig(image_name)
    plt.show()

    accuraccies = cross_val_score(estimator = model, X= x_train, y=y_train, cv=10)
    print("Average Accuracies: ",np.mean(accuraccies))
    print("Standart Deviation Accuracies: ",np.std(accuraccies))
    print("Accuracy of NB Score: ", model.score(x_test,y_test))

    report = classification_report(y_true, y_pred)
    print(report)
# applying Gausssian Naïve Bayes
GNB = GaussianNB()
model_NB(model_nb = GNB, image_name = 'GNB')
# applying Multinomial Naïve Bayes
MNB = MultinomialNB()
model_NB(model_nb = MNB, image_name = 'MNB')




