# -*- coding: utf-8 -*-
"""
Created on Sun Jun  2 18:53:41 2024

@author: Fedaa Almomani
         Shefaa Mestarihi
         Saja Zreqat
"""
"""

import numpy as np
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
# Loading the dataset

data = pd.read_excel("C:\\Users\\user\Downloads\Online Retail.xlsx\Online Retail.xlsx")

# Showing the data

data.head()
data.columns
data.shape

# Checkign whether there is any null values of not
data.isnull().values.any()

# As the previous cell told us that there are some null values. So, let's find them!
data.isnull().sum()

#  Data Preprocessing
# Stripping extra spaces in the description
data['Description'] = data['Description'].str.strip()

# Dropping the rows without any invoice number
data.dropna(axis = 0, subset =['InvoiceNo'], inplace = True)
data['InvoiceNo'] = data['InvoiceNo'].astype('str')

# Dropping all transactions which were done on credit
data = data[~data['InvoiceNo'].str.contains('C')]

# Let's see the countries in our dataset
data.Country.unique()
# Splitting the data according to the region of transaction
# Transactions done in France
basket_France = (data[data['Country'] =="France"]
          .groupby(['InvoiceNo', 'Description'])['Quantity']
          .sum().unstack().reset_index().fillna(0)
          .set_index('InvoiceNo'))

# Defining the hot encoding function to make the data suitable

def hot_encode(x):
    if(x<= 0):
        return 0
    if(x>= 1):
        return 1

# Applying one hot encoding

basket_encoded = basket_France.applymap(hot_encode)
basket_France = basket_encoded

basket_France.head()

# Building the model
frq_items = apriori(basket_France, min_support = 0.1, use_colnames = True)

# Collecting the inferred rules in a dataframe
rules = association_rules(frq_items, metric ="lift", min_threshold = 1)
rules = rules.sort_values(['confidence', 'lift'], ascending =[False, False])

print(rules.head())