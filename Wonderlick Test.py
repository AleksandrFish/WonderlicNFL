# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 08:22:47 2020

@author: afisher
"""

import requests
import pandas as pd
import numpy as np
import re
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns
sns.reset_defaults()
sns.set(rc={'figure.figsize':(3,2)}, 
    style="white")

url = "https://iqtestprep.com/nfl-wonderlic-scores/"
page = requests.get(url)
soup = BeautifulSoup(page.content, 'html.parser')
rows=soup.find_all('tr')

    
list_rows = []
for row in rows:
    cells = row.find_all('td')
    str_cells = str(cells)
    clean = re.compile('<.*?>')
    clean2 = (re.sub(clean, '',str_cells))
    list_rows.append(clean2)


# Put in DataFrame
df = pd.DataFrame(list_rows)
df.drop(df.index[0], inplace=True)
df = df[0].str.split(',', expand=True)
df[0] = df[0].str.strip('[')
df[2] = df[2].str.strip(']')

# Column Labels
col_labels = soup.find_all('th')
all_header = []
col_str = str(col_labels)
cleantext2 = BeautifulSoup(col_str, "lxml").get_text()

for k in cleantext2.split("\n"):
    header = " ".join(re.findall(r"[a-zA-Z0-9]+", k))

def convert(string): 
    li = list(string.split(" ")) 
    return li    

header = convert(header)
df.columns = header

#Filter to QBs
df['Position'] = df['Position'].str.replace(' ', '')
df_qbs = df.loc[df['Position']=='QB']

# Descriptive Statistics
df['Score'] = pd.to_numeric(df['Score'], errors='ignore')
df.groupby('Position')['Score'].count()


fig, ax = plt.subplots()
plt.hist(df['Score'])
ax.set_title('Wonderlick Score Distribution')
ax.set_ylabel('#')
fig.tight_layout()

sns.set(style="whitegrid")
sns.boxplot(df['Score'])
sns.barplot(y='Position', x='Score', data=df)


# Qb Stats until 2016
qb = pd.read_csv(r"C:\Users\afisher\Documents\Python Code\Projects\NFL\QBStats_all.csv")

qb_clean=[]
for i in qb['qb']:
    x = re.findall('[A-Z][^A-Z]*', i)
    qb_clean.append(x)
    
pd.Series(qb_clean).str.len()
qb_names = pd.DataFrame(qb_clean)
qb_names.fillna(value=np.nan, inplace=True)

qb_names[1] = np.where(qb_names[1].isin(["O'", "Mc"]), qb_names[1]+ qb_names[2], qb_names[1])

qb = pd.concat([qb, qb_names], axis=1)
qb = qb.loc[qb[4].isna()]
qb['Player'] = qb[0] + qb[1]
qb = qb[['Player', 'rate']]

qb.sort_values('Player', inplace=True)

qbstats = qb.groupby('Player', as_index=False)['rate'].agg({'count':'size', 'rating':'mean'})
qbstats.dropna(inplace=True)
qbstats = qbstats.loc[qbstats['count'] > 15]

# Merged Dataset
wonder = pd.merge(qbstats, df_qbs, on='Player')
sns.regplot(x="Score", y="rating", data=wonder)
plt.xlabel("Wonderlic Score")
plt.ylabel("QB Rating")
plt.title("Does the Wonderlic \n Predict NFL QB Success?")

# Export
wonder.to_csv(r'C:\Users\afisher\Documents\Python Code\Projects\NFL\QB_Wonderlick.csv')
