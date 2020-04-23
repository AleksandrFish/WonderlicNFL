# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 08:22:47 2020

@author: afisher
"""

# Import Libraries
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

from scipy.stats import pearsonr

#####################
# From IQ Test Prep #
#####################

# Get Wonderlic Scores
def get_wonderlic(position):
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
    column_names = []
    header_cells = rows[0].find_all("th")
    for cell in header_cells:
        header = cell.text
        header = header.strip()
        header = header.replace("\n", " ")
        column_names.append(header) 
        
    df.columns = column_names
    
    # Convert to numeric
    df['Score'] = pd.to_numeric(df['Score'], errors='ignore')
    
    #Filter Position
    df['Position'] = df['Position'].str.replace(' ', '')
    df_qbs = df.loc[df['Position']==position]
    df_qbs.sort_values('Player', inplace=True)
    df_qbs.drop_duplicates('Player', inplace=True)
    return df_qbs

df_qbs = get_wonderlic("QB")
df_rbs = get_wonderlic("RB")

###############################
# From Pro-Football Reference #
###############################

def get_qb_stats(year):
    url = "https://www.pro-football-reference.com/years/"+str(year)+"/passing.htm"
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
    
    df2 = pd.DataFrame(list_rows)
    df2.drop(df2.index[0], inplace=True)
    df2 = df2[0].str.split(',', expand=True)

    
    # Column Labels
    column_names = []
    header_cells = rows[0].find_all("th")
    for cell in header_cells:
        header = cell.text
        header = header.strip()
        header = header.replace("\n", " ")
        column_names.append(header) 
    
    column_names.remove(column_names[0])   
    df2.columns = column_names
    df2['Player'] = df2['Player'].str.strip('[')
    df2['GWD'] = df2['GWD'].str.strip(']')
    
    # Remove missing
    df2 = df2.loc[df2['Player']!="]"]
    df2 = df2.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df2 = df2.loc[df2['Pos']=="QB"]
    
    # Convert to numeric
    df2 = df2.apply(pd.to_numeric, errors='ignore')
    df2['Year'] = year
    df2 =df2[['Player', 'Tm', 'Age', 'QBrec', 
              'Cmp%', 'Yds', 'TD', 'Int', 'Rate', 'Year']]
    return df2

# Get data for last 20 years
years = list(range(1990, 2020))
d = {}
for year in years:
    d[year] = get_qb_stats(year)

# Rename Dictionary
def rename_dict(multilevelDict):
    return {"df_"+str(key): (transform(value) if isinstance(value, dict) else value) for key, value in multilevelDict.items()}
d=rename_dict(d)

# Combine Dictionaries
qbdata = pd.concat(d.values(), ignore_index=True)

# Remove Special characters
qbdata['Player'] = qbdata['Player'].str.replace('[^\w\s]','')

# Get number of games and average QB rating
qbdata_avg = qbdata.groupby('Player', as_index=False)['Rate'].agg({'count':'size', 'Rate':'mean'})
qbdata_avg .drop_duplicates('Player', inplace=True)

# Remove QBs who only played one season
qbdata_avg2 = qbdata_avg.loc[qbdata_avg['count'] > 1]
qbdata_avg2.drop_duplicates('Player', inplace=True)

# Merged Dataset
wonder = pd.merge(qbdata_avg, df_qbs, on='Player')
sns.regplot(x="Score", y="Rate", data=wonder)
plt.xlabel("Wonderlic Score")
plt.ylabel("QB Rating")
plt.title("Does the Wonderlic \n Predict NFL QB Success?")
plt.show()

#calculate correlation coefficient
corr = pearsonr(wonder['Score'], wonder['Rate'])
corr = [np.round(c, 2) for c in corr]
#add the coefficient to your graph
text = 'r=%s, p=%s' % (corr[0], corr[1])
ax = sns.regplot(x="Score", y="Rate", data=wonder)
ax.text(35, 110, text, fontsize=8)
plt.xlabel("Wonderlic Score")
plt.ylabel("QB Rating")
plt.title("Does the Wonderlic \n Predict NFL QB Success?")
plt.show()

# Export
wonder.to_csv(r'C:\Users\afisher\Documents\Python Code\Projects\NFL\QB_Wonderlick.csv')


# Rookie Year
qbdata.sort_values(['Player', 'Year'], inplace=True)
qbdata.drop_duplicates('Player', inplace=True)
wonder = pd.merge(qbdata_avg, df_qbs, on='Player')
sns.regplot(x="Score", y="Rate", data=wonder)
plt.xlabel("Wonderlic Score")
plt.ylabel("QB Rating")
plt.title("Does the Wonderlic \n Predict NFL QB Success?")
plt.show()
