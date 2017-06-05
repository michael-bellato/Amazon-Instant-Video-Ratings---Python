# -*- coding: utf-8 -*-
"""
Created on Fri Jun 02 16:20:15 2017

@author: Michael
Extract data creates summary CSV for users and products.
CSV files must be created or downloaded to Analyze
Analyze plots a histogram for the number of users across the number of reviews 
            per user.
     ...plots a scatter plot of users who made 3 or more reviews for their 
            individual reviews and the respective product average review.
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
n, bins, patches = plt.hist(df_user.user_count,facecolor='b')
plt.xlabel('Num of Users')
plt.ylabel('Reviews made')
plt.title('Reviews per User')
plt.yscale('log')

# scatter single user ratings (with 3 or more ratings) by average rating of each product
qu_ind = np.array(df_user.user_count>=3)# qualifying user index
usern = sum(qu_ind)                     # number of qualifying users
prodn = sum(df_user.user_count[qu_ind]) # number of related ratings
sct_users = df_user[qu_ind]             # qualifying users
usin = sct_users.index                  # user index

us_rate = np.array([0])                 # pre-allocate user ratings
pr_rate = np.array([0])                 # pre-allocate product average
for ii in usin: # concatenate average user rating with each prod rating
    rever = sct_users.uni_users[ii]          # reviewer ID
    count = np.int(sct_users.user_count[ii]) # number of reviews - integer
    pre_ur = np.matlib.repmat(rever,count,1) # placeholder for users - repmat for logical
    prRev = df_all.rating[df_all.reviewerID==sct_users.uni_users[ii]]    
    prID =  df_all.productID[df_all.reviewerID==sct_users.uni_users[ii]] 
    urmat = np.matlib.repmat(prID,len(df_prod),1)         # user repmat for logical
    prmat = np.matlib.repmat(df_prod.uni_prods,count,1).T # prod repmat for logical
    prind = np.array(np.sum(urmat==prmat,1),dtype=bool)   # product index
    pre_pr = df_prod.prod_rating[prind]                   # placeholder for prods
    us_rate = np.concatenate((us_rate,prRev),axis=0)      # all users ratings
    pr_rate = np.concatenate((pr_rate,pre_pr), axis=0)    # product avg

# Scatter plot of Individual User rating by average product rating
plt.figure(2)
plt.plot(us_rate,pr_rate,"o")
plt.xlabel('Individual User Rating')
plt.ylabel('Average Product Rating')
plt.title('Ind rating by Prod rating')
plt.xlim(-1,6)
plt.ylim(-1,6)
































