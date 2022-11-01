import pandas as pd
import numpy as np
import statsmodels.api as sm
import scipy.optimize as sco
import scipy.interpolate as sci
np.random.seed(8888)
import matplotlib.pyplot as plt
import scipy.stats as scs
import matplotlib as mpl
import tensorflow as tf
import tensorflow_datasets as tfds



#Station 1 ETL
df_stock = pd.read_excel("/Users/phoebe/Desktop/FINS5545/option 1/ASX200top10.xlsx",sheet_name='Data',index_col=0,header=8,nrows=11,usecols=[0,1,2])
print(df_stock)

df_bloom = pd.read_excel("/Users/phoebe/Desktop/FINS5545/option 1/ASX200top10.xlsx",sheet_name='Bloomberg raw',index_col=0,header=[0,1])
print(df_bloom)

df_economics = pd.read_excel("/Users/phoebe/Desktop/FINS5545/option 1/Economic_Indicators.xlsx",sheet_name='Sheet1',index_col=0)
print(df_economics)

df_news = pd.read_json("/Users/phoebe/Desktop/FINS5545/option 1/news_dump.json")
print(df_news)

df_client = pd.read_excel("/Users/phoebe/Desktop/FINS5545/option 1/Client_Details.xlsx",sheet_name='Data',index_col=0)
print(df_client)



#Station 2 Feature Engineering


# Day_to_Day return(to see the increasing or decreasing trend )
Daily_stock = pd.DataFrame(index = df_bloom.index,columns = df_stock.iloc[:,0])

for n in df_bloom.index:
    for m in df_stock.iloc[:,0]:
        Daily_stock.loc[n][m] =df_bloom.loc[n][m]['DAY_TO_DAY_TOT_RETURN_GROSS_DVDS']

Daily_stock = Daily_stock.astype(float)
Daily_stock

Daily_stock = Daily_stock.drop(['AS51 Index'],axis=1)
Daily_stock

l = Daily_stock/100
ret = l
print(ret)


#whether risk is high#
ret.hist(bins=100, figsize=(9,6))
plt.show()
ret.plot(subplots=True, style='b', figsize=(12, 30), grid=True, fontsize=12);
plt.show()

print(ret.describe())
print(ret.mean() * 252)
print(ret.cov() * 252)
print(ret.corr())




# PX_LAST(closing price)

Last_stock = pd.DataFrame(index = df_bloom.index,columns = df_stock.iloc[:,0])

for n in df_bloom.index:
    for m in df_stock.iloc[:,0]:
        Last_stock.loc[n][m] = df_bloom.loc[n][m]['PX_LAST']

Last_stock=Last_stock.astype(float)

#log_returan#

rets = np.log(Last_stock.iloc[:,1:]/Last_stock.shift(1).iloc[:,1:])




for n in df_stock.iloc[1:,0]:
    
    plt.hist(rets[n].values.flatten(), bins=70, label='frequency')
    plt.grid(True)
    plt.xlabel('log-return')
    plt.ylabel('frequency')
    x = np.linspace(plt.axis()[0], plt.axis()[1])
    plt.plot(x, scs.norm.pdf(x, loc=0.05 / 50, scale=0.38 / np.sqrt(50)),
         'r', lw=2.0, label='pdf')
    plt.legend()
    plt.savefig("plot1.png")
    
    
    sm.qqplot(rets[n].values.flatten()[::500], line='s')
    plt.grid(True)
    plt.xlabel('theoretical quantiles')
    plt.ylabel('sample quantiles')
    plt.savefig("plot2.png")


    
for n in df_stock.iloc[1:,0]:
   
    sm.qqplot(rets[n].values.flatten(), line='s')
    plt.grid(True)
    plt.xlabel('theoretical quantiles')
    plt.ylabel('sample quantiles')
    plt.savefig("plot2.png")



#see the trend of closing price#
rets.hist(bins=100, figsize=(9,6))
plt.show()
rets.plot(subplots=True, style='b', figsize=(12, 30), grid=True, fontsize=12);
plt.show()



print(rets.describe())
print(rets.mean() * 252)
print(rets.cov() * 252)
print(rets.corr())



#Station 3 : Model Design & Station 4 Implementation

import numpy as np
weights = np.random.random(10)
weights /= np.sum(weights)
print(weights)

# number of assets
noa = 10

print(ret.mean() * 252)
print(ret.cov() * 252)

#Setup random porfolio weights
weights = np.random.random(noa)
weights /= np.sum(weights)

#Derive Porfolio Returns & simulate various 2500x combinations
print(np.sum(ret.mean() * weights) * 252)
print(np.dot(weights.T, np.dot(ret.cov() * 252, weights)))

prets = []
pvols = []
for p in range (2500):
    weights = np.random.random(noa)
    weights /= np.sum(weights)
    prets.append(np.sum(ret.mean() * weights) * 252)
    pvols.append(np.sqrt(np.dot(weights.T,
                                np.dot(ret.cov() * 252, weights))))
prets = np.array(prets)  #portfolio return
pvols = np.array(pvols)  #portfolio volatility

plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets, c=prets / pvols, marker='o')
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()


def statistics(weights):
    weights = np.array(weights)
    pret = np.sum(ret.mean() * weights) * 252
    pvol = np.sqrt(np.dot(weights.T, np.dot(ret.cov() * 252, weights)))
    return np.array([pret, pvol, pret / pvol])

def min_func_sharpe(weights):
    return -statistics(weights)[2]

cons = ({'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(noa))
noa * [1. / noa,]

opts = sco.minimize(min_func_sharpe, noa * [1. / noa,], method='SLSQP',
                    bounds=bnds, constraints=cons)
print("***Maximization of Sharpe Ratio***")
#print(opts)
print(opts['x'].round(3))
print(statistics(opts['x']).round(3))

def min_func_variance(weights):
    return statistics(weights)[1] ** 2

optv = sco.minimize(min_func_variance, noa * [1. / noa,], method='SLSQP',
                    bounds=bnds, constraints=cons)
print("****Minimizing Variance***")
#print(optv)
print(optv['x'].round(3))
print(statistics(optv['x']).round(3))

def min_func_port(weights):
    return statistics(weights)[1]

bnds = tuple((0, 1) for x in weights)
trets = np.linspace(0.0, 0.25, 50)
tvols = []
for tret in trets:
    cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - tret},
            {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
    res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
                       bounds=bnds, constraints=cons)
    tvols.append(res['fun'])
tvols = np.array(tvols)

plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets,
            c=prets / pvols, marker='o')
# random portfolio composition
plt.scatter(tvols, trets,
            c=trets / tvols, marker='x')
# efficient frontier
plt.plot(statistics(opts['x'])[1], statistics(opts['x'])[0],
         'r*', markersize=15.0)
# portfolio with highest Sharpe ratio
plt.plot(statistics(optv['x'])[1], statistics(optv['x'])[0],
         'y*', markersize=15.0)
# minimum variance portfolio
plt.grid(True)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

ind = np.argmin(tvols)
evols = tvols[ind:]
erets = trets[ind:]
tck = sci.splrep(evols, erets)
def f(x):
    
    # Efficient frontier function (splines approximation). '''

    return sci.splev(x, tck, der=0)
def df(x):
    
    #''' First derivative of efficient frontier function. '''
    return sci.splev(x, tck, der=1)

def equations(p, rf=0.01):
    eq1 = rf - p[0]
    eq2 = rf + p[1] * p[2] - f(p[2])
    eq3 = p[1] - df(p[2])
    return eq1, eq2, eq3

opt = sco.fsolve(equations, [0.01, 0.5, 0.15])
#print(opt)
#print(np.round(equations(opt), 6))

plt.figure(figsize=(8, 4))
plt.scatter(pvols, prets,
            c=(prets - 0.01) / pvols, marker='o')
# random portfolio composition
plt.plot(evols, erets, 'g', lw=4.0)
# efficient frontier
cx = np.linspace(0.0, 0.3)
plt.plot(cx, opt[0] + opt[1] * cx, lw=1.5)
# capital market line
plt.plot(opt[2], f(opt[2]), 'r*', markersize=15.0)
plt.grid(True)
plt.axhline(0, color='k', ls='--', lw=2.0)
plt.axvline(0, color='k', ls='--', lw=2.0)
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.show()

cons = ({'type': 'eq', 'fun': lambda x:  statistics(x)[0] - f(opt[2])},
        {'type': 'eq', 'fun': lambda x:  np.sum(x) - 1})
res = sco.minimize(min_func_port, noa * [1. / noa,], method='SLSQP',
                   bounds=bnds, constraints=cons)
print("***Optimal Tangent Portfolio***")
print(res['x'].round(3))



#Trading news

# USE THIS DATASET TO TRAIN YOUR MODEL FOR THIS TASK
dataset, info = tfds.load('imdb_reviews/subwords8k', with_info=True,
                          as_supervised=True)
train_dataset, test_dataset = dataset['train'], dataset['test']

# SUPPORT FUNCTIONS
def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    plt.show()

def pad_to_size(vec, size):
    zeros = [0] * (size - len(vec))
    vec.extend(zeros)
    return vec

def sample_predict(sample_pred_text, pad):
    encoded_sample_pred_text = encoder.encode(sample_pred_text)
    if pad:
        encoded_sample_pred_text = pad_to_size(encoded_sample_pred_text, 64)
    encoded_sample_pred_text = tf.cast(encoded_sample_pred_text, tf.float32)
    predictions = model.predict(tf.expand_dims(encoded_sample_pred_text, 0))
    return (predictions)


# START BUILDING YOUR ML MODEL
encoder = info.features['text'].encoder

BUFFER_SIZE = 10000
BATCH_SIZE = 64

train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.padded_batch(BATCH_SIZE, ([-1],[]))
test_dataset = test_dataset.padded_batch(BATCH_SIZE, ([-1],[]))

# DEFINE THE MODEL HERE
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(encoder.vocab_size, 64),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer=tf.keras.optimizers.Adam(1e-4),
              metrics=['accuracy'])
history = model.fit(train_dataset, epochs=5,
                    validation_data=test_dataset,
                    validation_steps=30)
test_loss, test_acc = model.evaluate(test_dataset)

print('Test Loss: {}'.format(test_loss))
print('Test Accuracy: {}'.format(test_acc))

# HERE YOU RUN TRAINED MODEL (TRAINED ON MOVIES) ACROSS YOUR NEWS SNIPPETS
sentiment_index = []
for row in df_news['Headline']:
    sentiment_index.append(sample_predict(row, pad=True)[0])

# RESULTS SENTIMENT INDEX AND MODEL ACCURACY MEASURES
print(sentiment_index)
plot_graphs(history, 'accuracy')
plot_graphs(history, 'loss')
df1 = pd.DataFrame(sentiment_index)
df1.to_json('sentiment_index.json')

# PROCESS SENTIMENT DATA IN AGGREGATE TERMS
df2 = pd.read_json('sentiment_index.json')
print(df2.sort_index())

sentiment=pd.read_json('sentiment_index.json')
df1=pd.read_json("/Users/phoebe/Desktop/FINS5545/option 1/news_dump.json")
df2=pd.concat([df1,sentiment],axis=1,sort=False)
df2=df2.rename(columns={0:"Sentiment"})
plt.scatter(df2['Equity'],df2['Sentiment'])
plt.show()









