# -*- coding: utf-8 -*-
"""
Created on Sat Jun 10 10:05:48 2017

@author: Michael
"""

# Load Review file
# AmazonRev

import numpy as np
import pandas as pd
import numpy.matlib
import matplotlib.pyplot as plt

temp = 'C:\Users\Michael\Documents\DS_Projects\%s.csv'
# path = temp % 'AmaRevTest1k'
path = temp % 'ratings_Amazon_Instant_VideoTest'  # Load 500,000+ Instant Video Reviews
path_user = temp % 'UserSumm'                     # Load Summary of Users - avg rating
path_prod = temp % 'ProdSumm'                     # Load Summary of product - avg rating
# struct - userID; ProductID; Rating; time (unix)
df_all = pd.read_csv(path)                    # 500,000+ Instant Video Reviews
df_user = pd.read_csv(path_user)              # Summary of Users - avg rating
df_prod = pd.read_csv(path_prod)              # Summary of product - avg rating

# plot the average product rating by the user's average rating
sortn = np.sort(df_user.user_count)           # sort users by number of reviews 
threshind = round(len(sortn)*.90)             # create threshold for top 10% of reviewers
threshn = sortn[threshind]                    # identify threshold place
topind = df_user.user_count>threshn           # select users above threshold
# top_vals = np.array(df_user.user_count)[topind]

# Histogram of number of reviews per user
# rescale to log axis
plt.figure(1)
n, bins, patches = plt.hist(df_user.user_count,facecolor='r')
plt.xlabel('Reviews Made')
plt.ylabel('Number of Users')
plt.title('Reviews per User')       
plt.yscale('log')                   # Large volume of 1-review users, log scale
plt.savefig('RevXUser.png')

# scatter single user ratings (with 3 or more ratings) by average rating of each product
qu_ind = np.array(df_user.user_count>=2)# qualifying user index
usern = sum(qu_ind)                     # number of qualifying users
prodn = sum(df_user.user_count[qu_ind]) # number of related ratings
sct_users = df_user[qu_ind]             # qualifying users
usin = sct_users.index                  # user index

us_rate = np.array([0])                 # pre-allocate user ratings
pr_rate = np.array([0])                 # pre-allocate product average
us_bias = np.array([0])                 # pre-allocate user bias
us_adjRev =  np.array([0])              # pre-allocate adjusted review
pr_adjDiff =  np.array([0])              # pre-allocate adjusted product
for ii in usin: # concatenate average user rating with each prod rating
    prRev = df_all.rating[df_all.reviewerID==sct_users.uni_users[ii]]
    rever = sct_users.uni_users[ii]          # reviewer ID
    count = np.int(sct_users.user_count[ii]) # number of reviews - integer
    pre_ur = np.matlib.repmat(rever,count,1) # placeholder for users - repmat for logical  
    IndMuRev = np.squeeze(np.matlib.repmat(sum(prRev)/float(count),count,1)) 
    prID =  df_all.productID[df_all.reviewerID==sct_users.uni_users[ii]] 
    urmat = np.matlib.repmat(prID,len(df_prod),1)         # user repmat for logical
    prmat = np.matlib.repmat(df_prod.uni_prods,count,1).T # prod repmat for logical
    prind = np.array(np.sum(urmat==prmat,1),dtype=bool)   # product index
    pre_pr = df_prod.prod_rating[prind]                   # placeholder for prods
    pre_bias = np.reshape(np.mean(pre_pr-IndMuRev),1)     # placeholder for bias
    us_rate = np.concatenate((us_rate,IndMuRev),axis=0)   # users avg ratings
    pr_rate = np.concatenate((pr_rate,pre_pr), axis=0)    # product avg
    us_bias = np.concatenate((us_bias,pre_bias), axis=0)  # mean user bias
    usRevWbias = prRev+pre_bias
    prRevDiff = (pre_pr-pre_bias)-np.float64(prRev)
    us_adjRev = np.concatenate((us_adjRev,usRevWbias), axis=0)  # adjusted user review
    pr_adjDiff = np.concatenate((pr_adjDiff,prRevDiff), axis=0)  # estimated product review

xy = np.arange(1,5.1,.01)
# Scatter plot of Individual User rating by average product rating
plt.figure(2)
plt.plot(pr_rate[1:],us_rate[1:],"o",label="Individual Users")
plt.plot(xy,xy,'r',label="Equal Ratings")
plt.xlabel('Average User Rating')
plt.ylabel('Average Product Rating')
plt.title('Ratings - Ind Avg by Prod Avg')
plt.xlim(-1,6)
plt.ylim(-1,6)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.savefig('ProdXUser.png')

# Display Internal representations and Optimal Criterion 
# Parameters
sdv = 1;                                # Std Dev of Gaussian
mu = 0;                                 # Mu of Gaussian
xax = np.arange(-4,4.1,.1)              # Gaussian x-axis
a = 1/(sdv*np.sqrt(2*np.pi))            # Amplitude over x-axis
tg = np.exp((-1/2)*((xax-mu)/sdv)**2);  # Gauss over X-axis
gf = a*tg;                              # Amplitude times Gaussian
# Plot Gauss (X-axis computation allows for future manipulations)
badrev = (xax*1)-1;                     # X-axis for bad review (noise)
goodrev =(xax*1)+1;                     # X-axis for good review (signal)
optcrit = np.arange(0,.5,.1)            # Optimal Criterion
optxax = np.matlib.repmat(0,1,5)        # Optimal x-axis
plt.figure(3)
plt.plot(badrev,gf,color='red', label="Low Review Bias")
plt.plot(goodrev,gf,color='blue', label="High Review Bias")
plt.plot(np.squeeze(optxax),optcrit,color='black',label="Optimal Criterion")
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.xlim(-4,4)
plt.ylim(0,.5)
plt.title('Internal Representation of Product Quality')
plt.xlabel('Cognitive Bias')
plt.ylabel('Probability Density')
plt.savefig('CogBias.png')

# Compute Bias - distance between individual and product average
biasDiff = us_bias                      # Positive = Low Bias; Negative = High Bias
plt.figure(4)
n, bins, patches = plt.hist(biasDiff,facecolor='g', bins=16)
plt.title('Difference between Individual and Product Mu')
plt.xlabel('Rating Difference')
plt.ylabel('Number of Users')
plt.xlim(-4,4)

negRev =  sum(biasDiff>=0)              # Total user with low Bias
posRev =  sum(biasDiff<=0)              # Total user with high Bias
neuRev =  sum(biasDiff==0)              # Total user with no Bias
bary = np.array([negRev,neuRev,posRev]) # bar y-axis
barx = np.arange(-1,2)                  # bar x-axis
plt.figure(5)
plt.bar(barx,bary)
plt.title('Positive, Neutral, and Negative Biases')
plt.xlabel('Bias Class')
plt.ylabel('Number of Users')
plt.savefig('PosNNeg.png')

normn = n/max(n)                        # normalize to 1 for Internal Representation
normbin = np.zeros(len(bins)-1)         # Pre-allocate bin centers
for i in np.arange(len(normbin)):
    normbin[i] = (bins[i]+bins[i+1])/2  # calculate new bin locs

# Histogram of Internal Representations and Observed Bias
plt.figure(6)
plt.bar(normbin[::-1]*-1,normn[::-1],facecolor='g',width=.4,label="# of Users vs Bias")
plt.plot(badrev,gf,color='red', label="Low Review Bias")
plt.plot(goodrev,gf,color='blue', label="High Review Bias")
plt.plot(np.squeeze(optxax),optcrit,color='yellow',label="Optimal Criterion")
plt.title('Internal User Bias')
plt.xlabel('Rating Bias')
plt.ylabel('Normalized Number of Users')
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.xlim(-4,4)
plt.savefig('InternalBias.png')

# Scatter plot of User rating with Bias by average product rating
plt.figure(7)
plt.plot(pr_rate[1:],us_adjRev[1:],"o",label="Adjusted Reviews")
plt.plot(xy,xy,'r',label="Equal Ratings")
plt.xlabel('Adjusted User Rating')
plt.ylabel('Average Product Rating')
plt.title('Adjusted Ratings - Ind Avg by Prod Avg')
plt.xlim(-1,6)
plt.ylim(-1,6)
plt.legend(bbox_to_anchor=(1.05,1),loc=2,borderaxespad=0.)
plt.savefig('ProdXUserAdj.png')

# Apply bias to average ratings to estimate user rating
plt.figure(8)
plt.hist(abs(pr_adjDiff))
plt.xlabel('Difference between Bias and Predicted Rating')
plt.ylabel('Number of Users')
plt.title('Absolute Value Difference')
plt.savefig('AbsDiff.png')

critDiff = np.arange(0,5.1,.1)                  # range of differences from Bias
perCorr = np.zeros(len(critDiff))               # percent correct
rateLen = len(pr_adjDiff[1:])                   # total possible correct
diffInd = np.arange(0,len(critDiff))            # index var
for j in diffInd:
    corrRate = pr_adjDiff[1:]<=critDiff[j]
    perCorr[j] = sum(corrRate)/float(rateLen)
    
plt.figure(9)
plt.plot(critDiff,perCorr)
plt.xlabel('Predicted Rating Difference')
plt.ylabel('Percent Correct')
plt.title('Absolute Value Difference')
plt.xlim(0,5)
plt.ylim(0,1.1)
plt.savefig('ModelPred.png')









