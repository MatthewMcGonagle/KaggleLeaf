import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier

from random import shuffle
from mpl_toolkits.mplot3d import Axes3D

#pd.options.display.float_format = '{:,.3f}'.format
myformat = '{:.3f}'.format
seperator = '\n-------------------------\n'

ratios_df = pd.read_csv('ratiostest.csv')
train_df = pd.read_csv('Data/train.csv')
# train_df = train_df.loc[:, ['id', 'species']]

train_df = train_df.set_index('id')
print(seperator, 'Training Data Set Info')
print(train_df.info())
print(train_df[:10])

ratios_df = ratios_df.set_index('id')
print(seperator, 'Ratios Data Set Info')
print(ratios_df.info())
print(ratios_df[:10])

train_df = pd.concat([train_df, ratios_df], axis = 1, join_axes = [train_df.index])
print(seperator, 'After adding ratios column, training data is\n', train_df[:10])

# Now we need to encode the species category

le = LabelEncoder().fit(train_df.species)
train_df['species'] = le.transform(train_df.species)

# Now seperate data for cross validation

sss = StratifiedShuffleSplit(train_df['species'].values, 10, test_size = 0.3, random_state = 17)
for train_i, test_i in sss:
    # X_train, X_test = train_df.drop('species', axis = 1).values[train_i], train_df.drop('species', axis = 1).values[test_i]
    # X_train_noratio = train_df.drop(['species', 'isopratio'], axis = 1).values[train_i]
    # X_test_noratio =  train_df.drop(['species', 'isopratio'], axis = 1).values[test_i]
    # y_train, y_test = train_df['species'].values[train_i], train_df['species'].values[test_i]
    train_index = train_i
    test_index = test_i

test_df = train_df.iloc[test_index]
train_df = train_df.iloc[train_index]
y_train = train_df['species'].values
y_test = test_df['species'].values
print(seperator, 'After cross validation separation, description of the training data = \n', train_df.describe())
print('Now, the test data = \n', test_df.describe())

# Order the species by increasing mean isoperimetric ratio

traingroups = train_df.groupby('species').mean()
traingroups = traingroups.sort_values(by = 'isopratio', ascending = 1)
print(seperator, 'After grouping by species and reordering, traingroups = \n', traingroups[:10])
speciesorder = traingroups.index.values
print('Order of species is\n', speciesorder[:10])

train_df['species'] = train_df['species'].astype('category')
train_df['species'] = train_df['species'].cat.set_categories(speciesorder, ordered = True)

# Plot species on x-axis and ratios on y-axis. Categories appear in the order we specified above.
# Therefore, they appear in order of increasing mean ratio.

print(seperator, 'Training Data indices after attaching ordering\n', train_df.index[:10])
ax = sns.stripplot(x = 'species', y = 'isopratio', data = train_df)
plt.show()

# Now do PCA of data vectors

def getcollist(name, ncomponents):
    return [name + str(i + 1) for i in range(ncomponents)]

def keepncomponents(df, collist, ncomponents):
    df = df.drop( collist[ncomponents:], axis = 1 )
    collist = collist[:ncomponents]
    return df, collist

def normalizeattribute(df, collist):
    scaler = MinMaxScaler()
    scaler.fit( df[collist[0]] ) 
    df[collist] = scaler.transform(df[collist]) 
    return df

pca = decomposition.PCA()
ncomponents = {} # Dictionary for how many components kept for each type of vector

variance_ratios = {}
# Do PCA for each attribute, throw away unwanted components

for name in ['texture', 'shape', 'margin']:
    collist = getcollist(name, 64)
    pca.fit(train_df[collist])
    variance_ratios[name] = pca.explained_variance_ratio_
    train_df[collist] = pca.transform( train_df[collist] )
    test_df[collist] = pca.transform( test_df[collist] )

# Graph variance ratios of PCA for each attribute
for series in variance_ratios.values():
    plt.plot(series)
plt.legend(variance_ratios.keys(), loc = 'upper right')
plt.show()

# Number of PCA components to keep for each attribute
ncomponents['texture'] = 4
ncomponents['shape'] = 2
ncomponents['margin'] = 6

for name in ['texture', 'shape', 'margin']:
    collist = getcollist(name, 64)
    train_df, collist = keepncomponents(train_df, collist, ncomponents[name])
    train_df = normalizeattribute(train_df, collist)
    test_df, collist = keepncomponents(test_df, collist, ncomponents[name])
    test_df = normalizeattribute(test_df, collist)
 
print(seperator, 'After PCA, head of train_df is\n', train_df[:10])

# Normalize isoperimetric ratios

isopscaler = MinMaxScaler()
train_df['isopratio'] = isopscaler.fit_transform(train_df['isopratio'])
test_df['isopratio'] = isopscaler.transform(test_df['isopratio'])

# Now setup K Nearest Neighbors Classifier

predictioncols = ['ntexture', 'nshape', 'nmargin', 'num_nbs', 'with_ratio', 'accuracy', 'logloss']
predictions_df = pd.DataFrame(columns = predictioncols) 
componentcols = []
noutput = 2
npcas = [(ntexture, nshape, nmargin) 
    for ntexture in range(1,ncomponents['texture'] + 1) 
    for nshape in range(1,ncomponents['shape'] + 1) 
    for nmargin in range(1,ncomponents['margin'] + 1) 
    ]

for ntexture, nshape, nmargin in npcas: 
    componentcols = getcollist('texture', ntexture) 
    componentcols.extend(getcollist('shape', nshape))
    componentcols.extend(getcollist('margin', nmargin))

    X_train_with = train_df[componentcols + ['isopratio']].values
    X_test_with = test_df[componentcols + ['isopratio']].values
    X_train_without = train_df[componentcols].values
    X_test_without = test_df[componentcols].values

    for n_neighbors in range(1,10):
        clf = KNeighborsClassifier(n_neighbors)
        clf.fit(X_train_with, y_train)
        testpredictions = clf.predict(X_test_with)
        probpredictions = clf.predict_proba(X_test_with)
    
        # Accuracy and LogLoss
        acc = accuracy_score(y_test, testpredictions)
        ll = log_loss(y_test, probpredictions)
        newrow = [ntexture, nshape, nmargin, n_neighbors, 'with', acc, ll]
        newrow = pd.DataFrame([newrow], columns = predictioncols)
        predictions_df = predictions_df.append(newrow, ignore_index = True)
        
    
        # Do without ratios
        clf.fit(X_train_without, y_train)
        testpredictions = clf.predict(X_test_without)
        probpredictions = clf.predict_proba(X_test_without)
    
        # Accuracy and LogLoss for no ratio
        acc= accuracy_score(y_test, testpredictions)
        ll = log_loss(y_test, probpredictions)
        newrow = [ntexture, nshape, nmargin, n_neighbors, 'without', acc, ll]
        newrow = pd.DataFrame([newrow], columns = predictioncols)
        predictions_df = predictions_df.append(newrow, ignore_index = True)

predictions_df = predictions_df.groupby(['ntexture', 'nshape', 'nmargin', 'num_nbs', 'with_ratio']).mean()
predictions_df = predictions_df.unstack(4) 
predictions_df['logloss', 'improvement'] = predictions_df['logloss','without'] - predictions_df['logloss','with']

print(seperator, 'Summary of accuracies of K Nearest Neighbors For With and Without Isoperimetric Ratios:\n', predictions_df.applymap(myformat)) 

best_df = predictions_df[predictions_df['logloss', 'with'] < 1.0]
print(seperator, 'Summary of logloss less than 1.0\n', best_df.applymap(myformat) )
best_df = best_df.sort_values( [('logloss', 'with')] )
print(seperator, 'Now sorted by logloss with isoperimetric ratios\n', best_df.applymap(myformat) )

# Make 3d scatterplot
colordict = {}
for i in range(len(speciesorder)):
    colordict[speciesorder[i]] = i

varstoplot = ['texture1', 'texture2', 'texture3']
plotcolors = train_df['species'].apply(lambda x: colordict[x]) 
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection = '3d' )
ax2.scatter(train_df[varstoplot[0]], train_df[varstoplot[1]], train_df[varstoplot[2]], c = plotcolors)
plt.show()

# Now make pair plot
varstoplot = ['texture1', 'texture2', 'texture3']
plot_df = train_df[varstoplot + ['species'] ]
plot_df.sort_values(by = 'species')
ax = sns.pairplot(plot_df, hue = 'species', vars = varstoplot) 
plt.show()
