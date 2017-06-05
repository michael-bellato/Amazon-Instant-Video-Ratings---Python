# -*- coding: utf-8 -*-
"""
Created on Sat May 20 17:37:21 2017

@author: Michael Bellato
Amazon review test script
"""
# Load Review test file
# AmaTest1k

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

temp = 'C:\Users\Michael\Documents\DS_Projects\%s.csv'
# path = temp % 'AmaRevTest1k'
path = temp % 'ratings_Amazon_Instant_Video100k'

# struct - userID; ProductID; Rating; time (unix)
df = pd.read_csv(path)

# calc mean of all products
muprod = sum(df.rating) / float(len(df.rating))

# calc mean of each product
uni_prods = np.unique(df.productID)             # identify unique products
prodArr = np.array([df.productID])              # convert product list for logical

muProdRating = np.zeros((len(uni_prods),1))     # pre-allocate mu product rating
prodCount = np.zeros((len(uni_prods),1))        # pre-allocate product count
for i in np.arange(len(uni_prods)):             # for i throuh all of unique products
   prod = uni_prods[i]                          # select ith product ID
   prodInd = np.squeeze((prodArr == prod).T)    # logical index for prod ratings
   muProdRating[i] = np.mean(df.rating[prodInd])# calc mean prod rating
   prodCount[i] = sum(prodInd)                  # store occurances of product
   
# calc mean of Users
uni_users = np.unique(df.reviewerID)            # identiry unique users
userArr = np.array([df.reviewerID])             # convert user list for logical

muUserRating = np.zeros((len(uni_users),1))     # pre-allocate mu user rating
userCount = np.zeros((len(uni_users),1))        # pre-allocate user count
for ii in np.arange(len(uni_users)):            # for i through all of unique users
   user = uni_users[ii]                         # select iith user ID
   userInd = np.squeeze((userArr == user).T)    # logical index for user ratings
   muUserRating[ii] = np.mean(df.rating[userInd])# calc mean user ratings
   userCount[ii] = sum(userInd)                 # store number of ratings

####
sprodCount = np.squeeze(prodCount)
suserCount = np.squeeze(userCount)
smuProdRating = np.squeeze(muProdRating) 
smuUserRating = np.squeeze(muUserRating)

dfUniUsers = pd.DataFrame({'uni_users':uni_users})
dfmuUserRating = pd.DataFrame({'user_rating':smuUserRating})
dfUserCount = pd.DataFrame({'user_count':suserCount})
user_df = pd.concat([dfUniUsers,dfmuUserRating,dfUserCount],axis =1)

dfUniProds = pd.DataFrame({'uni_prods':uni_prods})
dfmuProdRating = pd.DataFrame({'prod_rating':smuProdRating})
dfProdCount = pd.DataFrame({'prod_count':sprodCount})
prod_df = pd.concat([dfUniProds,dfmuProdRating,dfProdCount],axis =1)

# creat new files
user_df.to_csv('UserSumm100k.csv')

prod_df.to_csv('ProdSumm100k.csv')






