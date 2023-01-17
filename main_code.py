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
    df_melted = pd.melt(df,id_vars='class',
                          var_name='Attributes',
                          value_name='Value')

    plt.figure()
    sns.boxplot(x='Attributes',y='Value',hue='class',data=df_melted)
    plt.xticks(rotation=75)
    plt.title('box plot before Standardization') 
    plt.savefig('box plot before Standardization')
    plt.show()

cwd = os.getcwd()
df = pd.read_csv('magic04.data', sep =',')
df_info(df)

plt.figure()
plt.bar(x='g', height= len(df[df['class']=='g']), label = 'Class g')
plt.bar(x='h', height= len(df[df['class']=='h']), label = 'class h')
plt.xlabel('class')
plt.ylabel('count')
plt.title('Count g & h')
plt.legend()
plt.savefig('count')
plt.show()

df_corr = df.corr()
plt.figure()
sns.clustermap(df_corr, annot = True, figsize=(16,10))
plt.title('Correlation between Attributes', fontdict={'fontsize':'14'})
plt.savefig('Correlation between Attributes')
plt.show()

box_plot(df = df, image_name = 'box plot before Standardization')

# df_melted = pd.melt(df,id_vars='class',
#                       var_name='Attributes',
#                       value_name='Value')

# plt.figure()
# sns.boxplot(x='Attributes',y='Value',hue='class',data=df_melted)
# plt.xticks(rotation=75)
# plt.title('box plot before Standardization') 
# plt.savefig('box plot before Standardization')
# plt.show()

x = df.drop(['class'], axis = 1)
y = df['class']
col = x.columns.tolist()
otl = LocalOutlierFactor()
pred = otl.fit_predict(x)

x_score = otl.negative_outlier_factor_
score = pd.DataFrame()
score['score'] = x_score

threshold = -1.5
outlier_index = score[score['score']<threshold].index.tolist()

x = x.drop(outlier_index)
y = y.drop(outlier_index).values

x_train, x_test,  y_train, y_test = train_test_split(x, y, 
                                                    test_size= 0.25
                                                    , random_state=42)

sc = MinMaxScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)
x_df =  pd.DataFrame(x_train, columns=col)
x_df['class'] = y_train


box_plot(df = x_df, image_name = 'box plot After Standardization')
# x_melted = pd.melt(x_df,id_vars='class',
#                       var_name='Attributes',
#                       value_name='Value')
# plt.figure()
# sns.boxplot(x='Attributes',y='Value',hue='class',data=x_melted)
# plt.xticks(rotation=75)
# plt.title('box plot After Standardization') 
# plt.savefig('box plot After Standardization')
# plt.show()

GNB = GaussianNB()

model_gnb = GNB.fit(x_train, y_train)
y_pred_gnb = model_gnb.predict(x_test)
y_true = y_test

cm = confusion_matrix(y_true, y_pred_gnb)


label = ['g','h']
plt.figure()
sns.heatmap(cm, annot= True, fmt='.0f', xticklabels = label
            , yticklabels = label)
plt.title("Confusion matrix for GNB")
plt.savefig('GNB')
plt.show()


accuraccies = cross_val_score(estimator = GNB, X= x_train, y=y_train, cv=10)
print("Average Accuracies: ",np.mean(accuraccies))
print("Standart Deviation Accuracies: ",np.std(accuraccies))
print("Accuracy of NB Score: ", GNB.score(x_test,y_test))



report = classification_report(y_true, y_pred_gnb)
print(report)


MNB = MultinomialNB()
model_mnb = MNB.fit(x_train, y_train)
y_pred_mnb = model_mnb.predict(x_test)

cm_mnb = confusion_matrix(y_true, y_pred_mnb)

label = ['g','h']
plt.figure()
sns.heatmap(cm_mnb, annot= True, fmt='.0f', xticklabels = label
            , yticklabels = label)
plt.title("Confusion matrix for MNB")
plt.savefig('MNB')
plt.show()

accuraccies = cross_val_score(estimator = MNB, X= x_train, y=y_train, cv=10)
print("Average Accuracies: ",np.mean(accuraccies))
print("Standart Deviation Accuracies: ",np.std(accuraccies))
print("Accuracy of NB Score: ", MNB.score(x_test,y_test))

report = classification_report(y_true, y_pred_mnb)
print(report)


