import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn import decomposition
from random import shuffle
from mpl_toolkits.mplot3d import Axes3D

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

traingroups = train_df.groupby('species').mean()
traingroups = traingroups.sort_values(by = 'isopratio', ascending = 1)
print(seperator, 'After grouping by species and reordering, traingroups = \n', traingroups[:10])
speciesorder = traingroups.index.values
print('Order of species is\n', speciesorder[:10])

train_df['species'] = train_df['species'].astype('category')
train_df['species'] = train_df['species'].cat.set_categories(speciesorder, ordered = True)

# Plot species on x-axis and ratios on y-axis. Categories appear in the order we specified above.
# Therefore, they appear in order of increasing mean ratio.

print(seperator, 'Training Data set after attaching ordering\n', train_df[:10])
ax = sns.stripplot(x = 'species', y = 'isopratio', data = train_df)
plt.show()

# Now do PCA on texturedata

print(seperator, 'Look at some texture columns')
print(train_df.loc[:10, ['texture1', 'texture2']] )
texturecollist = ['texture' + str(i+1) for i in range(64)]

# Combine texture columns into a column of lists 

# Consolidation of columns into one column of arrays is unnecessary. The
# following two lines of commented code can probably be removed. 
# train_df['texturevect'] = train_df[texturecollist].values.tolist()
# print(train_df.loc[:10, 'texturevect'])

# Now do PCA decomposition on textures
# pca = decomposition.PCA(n_components = 2)
# texture_matrix = train_df[texturecollist].values
# pca.fit( train_df[texturecollist] )
# texture_matrix = pca.transform( texture_matrix )
# print(texture_matrix[:3] )
# for i in range(len(texture_matrix[0,:])):
#     colname = 'texture_pca' + str(i+1)
#     train_df[colname] = texture_matrix[:, i]
# 
# print(seperator, 'Result of pca analysis of textures = \n', train_df.loc[:10, ['texture_pca1', 'texture_pca2']] )
 
##################################################

def do_pca_of(df, name, ncomponents):
    noutput = 2
    collist = [name + str( i + 1) for i in range(ncomponents) ]

    pca = decomposition.PCA(n_components = noutput)
    pca.fit( df[collist] )
    df[collist[:noutput]] = pca.transform( df[collist] )
    df = df.drop(collist[noutput:], axis = 1)
    return df

# Now do PCA of shape vectors

train_df = do_pca_of(train_df, 'texture', 64)
train_df = do_pca_of(train_df, 'shape', 64)
train_df = do_pca_of(train_df, 'margin', 64)

print(seperator, 'After PCA, head of train_df is\n', train_df.head())

### This is code for randomly reordering the species category as an attempt to
### eliminate any correlation between graph coloring and the isoperimetric ratio.
### Had problems getting it to change the order of the coloring on the seaborn graphs.
### However, it is commented out right now, because the pairwise plot below for PCA values
### of margin, shape, and texture benefits from this correlation.
# Randomly shuffle order of species
# shuffle(speciesorder)
# print('New random order of species is\n', speciesorder[:10] )
# train_df['species'] = train_df['species'].cat.reorder_categories(speciesorder, ordered = True)
# train_df.sort_values(by = 'species')

colordict = {}
for i in range(len(speciesorder)):
    colordict[speciesorder[i]] = i

plotcolors = train_df['species'].apply(lambda x: colordict[x]) 
fig2 = plt.figure()
ax2 = fig2.add_subplot(111, projection = '3d' )
ax2.scatter(train_df['isopratio'], train_df['texture1'], train_df['texture2'], c = plotcolors)
plt.show()

# sns.set()
plt.close()
plot_df = train_df[['species', 'margin1', 'texture1', 'shape1']]
plot_df.sort_values(by = 'species')
ax = sns.pairplot(plot_df, hue = 'species')
plt.show()
