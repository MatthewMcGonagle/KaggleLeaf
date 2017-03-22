import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from sklearn import decomposition
from sklearn.preprocessing import (LabelEncoder, MinMaxScaler, PolynomialFeatures)
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, log_loss
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline 
from sklearn.linear_model import LinearRegression

from random import shuffle
from mpl_toolkits.mplot3d import Axes3D

#pd.options.display.float_format = '{:,.3f}'.format
myformat = '{:.3f}'.format

def myformat(x):
    if isinstance(x, str):
        return [x]
    else:
        return '{:.3f}'.format(x)

seperator = '\n' + ('-'*80) + '\n'

ratios_df = pd.read_csv('ratiostest.csv')
train_df = pd.read_csv('Data/train.csv')
# train_df = train_df.loc[:, ['id', 'species']]

train_df = train_df.set_index('id')
print(seperator, 'Training Data Set Info')
print(train_df.info())
print(train_df[:10].applymap(myformat))

ratios_df = ratios_df.set_index('id')
print(seperator, 'Ratios Data Set Info')
print(ratios_df.info())
print(ratios_df[:10].applymap(myformat))

train_df = pd.concat([train_df, ratios_df], axis = 1, join_axes = [train_df.index])
print(seperator, 'After adding ratios column, training data is\n', train_df[:10].applymap(myformat))

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
print(seperator, 'After cross validation separation, description of the training data = \n', train_df.applymap(myformat).describe())
print('After cross validation separation, description of the test data = \n', test_df.applymap(myformat).describe())

# Order the species by increasing mean isoperimetric ratio

traingroups = train_df.groupby('species').mean()
traingroups = traingroups.sort_values(by = 'isopratio', ascending = 1)
print(seperator, 'After grouping by species and reordering, traingroups = \n', traingroups.applymap(myformat)[:10])
speciesorder = traingroups.index.values
print('Order of species is\n', speciesorder[:10])

train_df['species'] = train_df['species'].astype('category')
train_df['species'] = train_df['species'].cat.set_categories(speciesorder, ordered = True)

# Plot species on x-axis and ratios on y-axis. Categories appear in the order we specified above.
# Therefore, they appear in order of increasing mean ratio.

print(seperator, 'Training Data indices after attaching ordering\n', train_df.applymap(myformat).index[:10])
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
    # train_df = normalizeattribute(train_df, collist)
    test_df, collist = keepncomponents(test_df, collist, ncomponents[name])
    # test_df = normalizeattribute(test_df, collist)
    scaler = MinMaxScaler()
    scaler.fit(train_df[collist[0]].to_frame())
    train_df[collist] = scaler.transform(train_df[collist])
    test_df[collist] = scaler.transform(test_df[collist])
    
 
print(seperator, 'After PCA, head of train_df is\n', train_df.applymap(myformat)[:10])

# Normalize isoperimetric ratios

isopscaler = MinMaxScaler()
train_df['isopratio'] = isopscaler.fit_transform(train_df['isopratio'].values.reshape(-1, 1))
test_df['isopratio'] = isopscaler.transform(test_df['isopratio'].values.reshape(-1, 1))

# Do Quadratic Fit for Texture Components based on surface.

quadfitX = ['texture1', 'texture2']
quadfity = ['texture3']
model = Pipeline([ ('polynomial_features', PolynomialFeatures(3)),
                   ('linear_regression', LinearRegression())
                 ])
model.fit(train_df[quadfitX], train_df[quadfity])
predictresults = model.predict(train_df[quadfitX])
for name, newvalues in zip(quadfity, predictresults.T):
   train_df[name + 'fit'] = newvalues 
predictresults = model.predict(test_df[quadfitX])
for name, newvalues in zip(quadfity, predictresults.T):
   test_df[name + 'fit'] = newvalues 

# Generate grid of fit
Xfit = np.arange(np.amin(train_df[quadfitX[0]].values) - 0.1,
                 np.amax(train_df[quadfitX[0]].values) + 0.1,
                 0.003)
Yfit = np.arange(np.amin(train_df[quadfitX[1]].values) - 0.1,
                 np.amax(train_df[quadfitX[1]].values) + 0.1,
                 0.003)

# Make 3d scatterplot and plot of fit
varstoplot = quadfitX + quadfity 
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection = '3d' )
ax2.scatter(train_df[varstoplot[0]], train_df[varstoplot[1]], train_df[varstoplot[2]], c = 'black')

Xfit, Yfit = np.meshgrid(Xfit, Yfit)
Zfit = np.dstack( (Xfit, Yfit) )
print(Zfit[:3,:4])
Zfit = [model.predict(x) for x in Zfit[:,:] ]
Zfit = np.array(Zfit)
ni, nj, nk = Zfit.shape
Zfit = Zfit.reshape((ni, nj))
print(Zfit.shape) 
print(Xfit.shape)
ax2.plot_wireframe(Xfit, Yfit, Zfit, rstride = 10, cstride = 10)
plt.show()

# Now project onto surface

allfit = np.dstack( (Xfit, Yfit, Zfit) )
print(allfit.shape)

minpos = []
for point in train_df[quadfitX + quadfity].values[:]:
    diff = allfit - point
    diff = np.linalg.norm(diff, axis = 2)
    newpos = np.unravel_index(np.argmin(diff), diff.shape)
    minpos.append(newpos)
print(minpos[0])
minpos = np.array(minpos)
projpoints = []
for i, j in minpos:
    projpoints.append(allfit[i, j])
projpoints = np.array(projpoints)

projdiff = train_df[quadfitX + quadfity].values - projpoints
projerror = []
for point in projdiff:
    size = np.linalg.norm(point)
    if point[2] > 0:
        projerror.append(size)
    else:
        projerror.append(-size)

train_df['texture5'] = projerror
ncomponents['texture'] = 5

projpoints = projpoints.T
for i in range(len(projpoints)):
    train_df['texture' + str(i+1)] = projpoints[i] 

# Now do for test data

minpos = []
for point in test_df[quadfitX + quadfity].values[:]:
    diff = allfit - point
    diff = np.linalg.norm(diff, axis = 2)
    newpos = np.unravel_index(np.argmin(diff), diff.shape)
    minpos.append(newpos)
print(minpos[0])
minpos = np.array(minpos)
projpoints = []
for i, j in minpos:
    projpoints.append(allfit[i, j])
projpoints = np.array(projpoints)

projdiff = test_df[quadfitX + quadfity].values - projpoints
projerror = []
for point in projdiff:
    size = np.linalg.norm(point)
    if point[2] > 0:
        projerror.append(size)
    else:
        projerror.append(-size)

test_df['texture5'] = projerror
ncomponents['texture'] = 5


projpoints = projpoints.T
print('projpoints shape = ', projpoints.shape)
for i in range(len(projpoints)):
    test_df['texture' + str(i+1)] = projpoints[i] 

# This is the quadratic fit based on a curve. Other section contains fit based on surface.
# # Do Quadratic Fit for Texture Components
# 
# quadn = {}
# quadn['texture'] = 2
# quadfitcols = ['texture2', 'texture3']
# newquadcols = [name + 'fit' for name in quadfitcols]
# print(newquadcols)
# model = Pipeline([ ('polynomial_features', PolynomialFeatures(2)),
#                    ('linear_regression', LinearRegression())
#                  ] )
# model.fit(train_df['texture1'].to_frame(), train_df[quadfitcols] )
# predictresults = model.predict(train_df['texture1'].to_frame())
# for newseries, newvalues in zip(newquadcols, predictresults.T[:]):
#     train_df[newseries] = newvalues
# predictresults = model.predict(test_df['texture1'].to_frame())
# for newseries, newvalues in zip(newquadcols, predictresults.T[:]):
#     test_df[newseries] = newvalues
# 
# for name, color in zip(quadfitcols, ['blue', 'black', 'purple']):
#     plt.scatter(train_df['texture1'].values, train_df[name].values, color = color )
#     plt.scatter(train_df['texture1'].values, train_df[name + 'fit'].values, color = 'red')
# plt.show()
# 
# # for name in quadfitcols: 
# #     train_df[name] = train_df[name] - train_df[name + 'fit']
# #     test_df[name] = test_df[name] - test_df[name + 'fit']
# 
# # Project onto curve
# dx = 0.01
# xvalues = np.linspace(-2, 2, num = 400)
# xvalues = xvalues.reshape(len(xvalues), 1)
# yzvalues = model.predict(xvalues)
# allvalues = np.concatenate((xvalues, yzvalues), axis = 1)
# 
# minpos = []
# for point in train_df[['texture1'] + quadfitcols].values:
#    diff = allvalues - point
#    diff = np.linalg.norm(diff, axis = 1) 
#    minpos.append( np.argmin(diff) )
# minpos = np.array(minpos)
# 
# newpos = allvalues[minpos[:]]
# fitdiff = newpos - train_df[['texture1'] + quadfitcols].values
# train_df['texture2'] = np.linalg.norm(fitdiff, axis = 1)
# train_df['texture1'] = newpos.T[0] 
# # for name, j in zip(quadfitcols, range(1, len(quadfitcols) + 1)):
# #     train_df[name] = newpos[j] 
# 
# minpos = []
# for point in test_df[['texture1'] + quadfitcols].values:
#    diff = allvalues - point
#    diff = np.linalg.norm(diff, axis = 1) 
#    minpos.append( np.argmin(diff) )
# minpos = np.array(minpos)
# 
# newpos = allvalues[minpos[:]]
# fitdiff = newpos - test_df[['texture1'] + quadfitcols].values
# test_df['texture2'] = np.linalg.norm(fitdiff, axis = 1)
# test_df['texture1'] = newpos.T[0] 
# # for name, j in zip(quadfitcols, range(1, len(quadfitcols) + 1)):
# #     train_df[name] = newpos[j] 
# 
# # Now get info for computing arclength
# print('Linear Regression Coefficients Shape = ', model.named_steps['linear_regression'].coef_.shape)
# quadcoeff = model.named_steps['linear_regression'].coef_.T # Shape is now (n_features, n_targets)
# arclengthparam = np.zeros(3)
# arclengthparam[2] = np.linalg.norm(quadcoeff[2])
# dotAB = np.dot(quadcoeff[2], quadcoeff[1])
# arclengthparam[1] = dotAB * 0.5 / arclengthparam[2]**2
# arclengthparam[0] = 1.0 + np.linalg.norm(quadcoeff[1])**2 - dotAB**2 * arclengthparam[2]**-2.0
# print('Before square root, arclengthparam[0] = ', arclengthparam[0])
# arclengthparam[0] = np.sqrt(arclengthparam[0])
# print('arclengthparam = \n', arclengthparam)
# 
# def antideriv(x): # This is the anti-deriv of sqrt(1 + x**2) which is used to compute arclength
#     result = x * np.sqrt(1 + x**2) + np.arcsinh(x)
#     return result * 0.5
# 
# def getarclength(x, arclengthparam):
#     u1 = arclengthparam[1] * 2 * arclengthparam[2] / arclengthparam[0]
#     u2 = u1 + x * 2 * arclengthparam[2] / arclengthparam[0]
#     result = antideriv(u2) - antideriv(u1)
#     result *= arclengthparam[0]**2 * 0.5 / arclengthparam[2]
#     return result
# 
# # Transform train_df['texture1'] into arclength along fitted curve
# train_df['texture1'] = train_df['texture1'].apply(lambda x: getarclength(x, arclengthparam) )    
# test_df['texture1'] = test_df['texture1'].apply(lambda x: getarclength(x, arclengthparam) )    
#     

# Scale on 'texture1'
# Need to renormalize texture features using 'texture1'
collist = getcollist('texture', ncomponents['texture'])
scaler = MinMaxScaler()
scaler.fit(train_df['texture1'].to_frame())
train_df[collist] = scaler.transform(train_df[collist])
test_df[collist] = scaler.transform(test_df[collist])

# Scale on total size of fitted texture features
# sizes = train_df[quadfitX + quadfity].values
# sizes = np.linalg.norm(sizes, axis = 1)
# scaler = MinMaxScaler()
# scaler.fit(sizes.reshape((len(sizes), 1)) )
# collist = getcollist('texture', ncomponents['texture'])
# train_df[collist] = scaler.transform(train_df[collist])
# test_df[collist] = scaler.transform(test_df[collist])

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
