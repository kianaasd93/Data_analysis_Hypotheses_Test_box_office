#!/usr/bin/env python
# coding: utf-8

# # Data analysis and Hypotheses Test for SussexBudgetProductions

# In[1]:


#We read our file into a dataframe then make a copy of the original.
import pandas as pd
import copy
myfile='movie_metadata.csv'

import numpy as np

from matplotlib import pyplot as plt
import pandas as pd

myfile = 'movie_metadata.csv'


# ### Data_wrangling

# During the data_wrangling process, I change the order of the columns so that were easier to read. Additionally, I shortened some of the column names to make them more concise. This helped to streamline the data and make it more efficient for analysis. I delete any columns that are not relevant to our analysis and can't make any changes on our profit.

# In[2]:


#I change the order of the columns so that were easier to read 
df2 = pd.read_csv(myfile,index_col=None)

new_order = ["movie_title", "genres", "director_name", "actor_1_name", "actor_2_name", "actor_3_name", "budget", "imdb_score", "num_critic_for_reviews", "num_voted_users", "num_user_for_reviews", "content_rating", "aspect_ratio", "title_year", "country", "language", "color", "duration", "gross", "plot_keywords", "facenumber_in_poster", "director_facebook_likes", "actor_1_facebook_likes", "actor_2_facebook_likes", "actor_3_facebook_likes", "cast_total_facebook_likes", "movie_facebook_likes", "movie_imdb_link"]


df = copy.deepcopy(df2)
df = df[new_order]
df.head(5)


# In[3]:


del df['movie_imdb_link']
del df["plot_keywords"]


# In[4]:


columns = {"movie_title" : "movie_title",
           "genres" : "genres",
           #"num_critic_for_reviews" : "reviews_num",
           "director_name" : "director",
           "actor_1_name" : "actor1",
           "actor_2_name" : "actor2",
           "actor_3_name" : "actor3",
           "budget" : "budget",
           "imdb_score" : "imdb_score",
           "num_critic_for_reviews" : "critic_reviews",
           "num_voted_users" : "vote",
           "dnum_user_for_reviews" : "user_reviews",
           "content_rating" : "content_rating",
           "aspect_ratio" : "aspect_ratio",
           "title_year" : "year",
           "country" : "country",
           "language" : "language",
           "color" : "colorful",
           "duration" : "duration",
           "gross" : "gross",
           #"plot_keywords" : "plot_keywords",
           "facenumber_in_poster" : "facenumber_in_poster",
           "director_facebook_likes" : "director_facebook_likes",
           "actor_1_facebook_likes" : "actor_1_facebook_likes",
           "actor_2_facebook_likes" : "actor_2_facebook_likes",
           "actor_3_facebook_likes" : "actor_3_facebook_likes",
           "cast_total_facebook_likes" : "cast_total_facebook_likes",
           "movie_facebook_likes" : "movie_facebook_likes"}

df.rename(columns=columns,inplace=True)

df.head(5)


# ### Data_Cleaning
# 

# During the data_cleaning process, I addressed the null values in the most important columns that I wanted to perform analysis on. I did not remove null values from all columns at once, since I did not want to lose any valuable data. Instead, I will address null values in other columns later. check for duplicates and remove any movies that appear more than once, created a new column called 'benefit', which is the most important factor in our analysis.  

# In[5]:


#I addressed the null values in the most important columns that I wanted to perform analysis on
# I did not remove null values from all columns at once
#since I did not want to lose any valuable data. 
#I believe that having more data will ultimately lead to more efficient analysis.
nun = df[df["budget"].isnull()].index

df.drop(nun,inplace=True)

df.reset_index(drop=True,inplace=True)

 


# In[6]:


nun2 = df[df["gross"].isnull()].index


df.drop(nun2,inplace=True)

df.reset_index(drop=True,inplace=True)

 


# In[7]:


nun3 = df[df["imdb_score"].isnull()].index


df.drop(nun3,inplace=True)
df.reset_index(drop=True,inplace=True)


# In[8]:


nun4 = df[df["genres"].isnull()].index

df.drop(nun4,inplace=True)

df.reset_index(drop=True,inplace=True)

 


# In[9]:


df["benefit"] = df['gross'] - df['budget']



# In[10]:


pd.unique(df.movie_title)
df.movie_title.value_counts()


# In[11]:


df['movie_title'] = df['movie_title'].drop_duplicates(keep='first')


# In[12]:


df.budget.mean()


# # Data_Analysis

# Since we're only interested in high-quality films, we looked only at films with good ratings to ensure that our analysis is based on those that are likely to be profitable. It appears that 'imdb score' and 'benefit' columns are related, plotting would be accurate way to visualize the relationship between variables.

# In[13]:


import seaborn as sns 
import matplotlib.pyplot as plt
df.plot.scatter('imdb_score','benefit',ylim=(6.502210e+06,9.502210e+07))


# In[14]:


import seaborn as sns 
import matplotlib.pyplot as plt
df.plot.scatter('cast_total_facebook_likes','benefit', ylim=(6.502210e+06,9.502210e+07), xlim=(0,60000))


# In[15]:


import seaborn as sns 
import matplotlib.pyplot as plt
df.plot.scatter('director_facebook_likes','benefit', ylim=(6.502210e+06,9.502210e+07), xlim=(0,600))


# In[16]:


import matplotlib.pyplot as plt 
import seaborn as sns
import numpy as np 
from statsmodels.distributions.empirical_distribution import ECDF
fig, axes = plt.subplots(nrows=2, ncols=5, sharey=True)

for i in range(0, 10):
    j = i + 1
    lower_bound = i
    upper_bound = j
    float_range = [k / 10.0 for k in range(i * 10, (j * 10) + 1)]
    
    filtered_data = df[df['imdb_score'].isin(float_range)]
    
    sns.histplot(data=filtered_data, x="imdb_score", ax=axes[i // 5, i % 5], kde=True, bins=20, stat="count")
    sns.kdeplot(data=filtered_data, x="imdb_score", ax=axes[i // 5, i % 5], color="k", linewidth=2)
    axes[i // 5, i % 5].set_title(f"IMDb Score {lower_bound}-{upper_bound}")


fig.set_figwidth(fig.get_figwidth() * 4.5)
fig.set_figheight(fig.get_figheight() * 2.5)

plt.tight_layout()
plt.show()


# See that the most frequently occurring IMDb ratings are between 5.5 and 8. Therefore, we analyze IMDb ratings that are greater than 5.5.

# Relationship between variables in heatmap.

# In[17]:


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numerical_columns].corr(method='pearson')
plt.figure(figsize=(10, 8))
sns.heatmap(data=correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f")
plt.show()


# In[18]:


numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
correlation_matrix = df[numerical_columns].corr(method="spearman")
plt.figure(figsize=(10, 8))
sns.heatmap(data=correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f")
plt.show()


# Since we have budget limitations and I apply the IMDb rating limit that I previously determined, it is better to apply them to perform a more thorough analysis

# In[19]:


#the mean of budget is 4.2434653 so our budget is less than a mean.
#it would be better to apply them in a way that will allow us to perform better.
max_budget = 1500000
min_imdb_score=5.5
budget_fix = df[df['budget']<=max_budget]


top_rated = budget_fix[budget_fix['imdb_score']>=min_imdb_score]
top_rated = top_rated.sort_values('budget',ascending=False)
##By looking at the describe() function, we obtain important information about our data, such as the minimum and maximum values for each column.
top_rated.describe()


# we obtain important information, such as the minimum and maximum values for each column.

# In[20]:


top_rated.head(5)


# I sorted the mean IMDB score, budget, gross, and benefit across films by genre sorted by benefit.

# In[21]:


#we will analyze the data to determine which genres have the highest benefit.
genreseries=top_rated.groupby('genres')[['imdb_score','budget', "gross", 'benefit']].mean()



top_rated_genre = genreseries.sort_values('benefit', ascending=False)


top_rated_genre.head(10)


# I will count the number of occurrences of each genre and determine which genres are most commonly represented.

# 
# 

# In[22]:


# function for seprating the genres
def update_genres(val):
    list = []
    
    splits = ["|"]
    for substr in splits:

        if substr in val:
            while substr in val:
                split_idx = val.find(substr)
                a = val[:split_idx].strip()
                list.append(a)

                val = val[split_idx+1 :]

            list.append(val)
        else:
            list.append(val)
            
    
    return (list)



# In[23]:


#Apply the def for our genres in top_rated
from collections import Counter

list_top_genres= [update_genres(i) for i in top_rated['genres']]


# In[24]:


#count how many times the genres appeared in top_rated 
top_list = []
listt = []
for item in list_top_genres :
    for items in item:
        if items not in listt:
            listt.append(items)
            item_count = sum(items in sublist for sublist in list_top_genres )

            print(f"{items} appears {item_count} times .")
            if item_count >=10:
                    top_list.append(items)
print(top_list)


# According to the top_list (genres in the highest benefits movies), we selecte our genre, use one or mix of genres. 
# We examine some columns to identify the unique values and how frequently they occur.

# In[25]:


#It is a good time to remove  null values.
# Assuming 'df' is your DataFrame
# It is a good time to remove null values.
import pandas as pd
for column in df.columns:
    null_indices = df[df[column].isnull()].index
    df.drop(null_indices, inplace=True)

# Reset the index after dropping rows
df.reset_index(drop=True, inplace=True)


# I categorize the data into R, PG-13, PG, G, X, NC-17 and Instead of removing any data, assign it to more relevant categories.

# In[26]:


pd.unique(df.content_rating)
df.content_rating.value_counts()


# In[27]:


#According to our conversation in the meeting about content_rating
#write a def that filter the content_rating and apply it
def update_content_rating(val):
    remove = [ "Not Rated","Unrated","Approved","M","Passed"]
    remove2 = ["GP"]
    for i in remove:
        if i in val:
           
            return "G"
    for i in remove2:
        if i in val:
            
            return "PG"   

    return val


df.content_rating = df.content_rating.apply(update_content_rating)

pd.unique(df.content_rating)
df.content_rating.value_counts()


# Selecting movies in the top genres to make informed decisions. Our attention will be towards movies with high benefits compared to the mean benefit. we go into specific details to identify additional features. For haveing the name of the actors and directors that are alive we put a limit for years.

# In[28]:


for column in top_rated.columns[:21]:
    nun = top_rated[top_rated[column].isnull()].index
    top_rated.drop(nun,inplace=True)
    top_rated.reset_index(drop=True,inplace=True)


# In[29]:


year = 1950
year_filter = top_rated[top_rated["year"] >= year]


# In[30]:


#Our attention will be towards movies with high benefits compared to the mean benefit
print(top_rated.benefit.mean())


# In[31]:


ok = 1000000

high_benefit = year_filter[year_filter["benefit"] >= ok]


# In[32]:


#We have our top genres in the top_list
top_genres = high_benefit[high_benefit['genres'].isin(top_list)]


# In[33]:


top_genres = top_genres.sort_values('benefit', ascending=False)


# In[34]:


top_genres.head(5)


# In[35]:


pd.unique(top_genres.colorful)
top_genres.colorful.value_counts()


# In[36]:


ax1 = top_genres[top_genres['colorful']=='Color'].plot(kind='scatter', x='imdb_score', y='benefit',xlim=(0,10),color='g')
ax2 = top_genres[top_genres['colorful']!='Color'].plot(kind='scatter', x='imdb_score', y='benefit',xlim=(0,10),color='r', ax=ax1)
ax2=ax1.legend(['Color', 'Black and White'])


# In[37]:


pd.unique(top_genres.director)
top_genres.director.value_counts()


# In[38]:


pd.unique(top_genres.actor1)
top_genres.actor1.value_counts()


# In[39]:


pd.unique(top_genres.actor2)
top_genres.actor2.value_counts()


# In[40]:


pd.unique(top_genres.actor3)
top_genres.actor3.value_counts()


# In[41]:


pd.unique(top_genres.content_rating)
top_genres.content_rating.value_counts()


# In[42]:


pd.unique(top_genres.aspect_ratio)
top_genres.aspect_ratio.value_counts()


# In[43]:


pd.unique(top_genres.country)
top_genres.country.value_counts()


# In[44]:


pd.unique(top_genres.language)
top_genres.language.value_counts()


# In[45]:


pd.unique(top_genres.facenumber_in_poster)
top_genres.facenumber_in_poster.value_counts()


# In[46]:


pd.unique(top_genres.duration)
top_genres.duration.value_counts()


# Based on the value counts, we selecte features for our movies, including colorful, English language, R_content_rating, 1.85_aspect_ratio, 0_facenumber_in_poster, and 81-100 duration. Considering the preferred genres (Comedy, Horror, and Drama leading the list), we have two options: selecting actors and directors from the highest-benefit movies or considering their Facebook likes, which we will cover later.

# # Testing

# In[47]:


# I get the idea form lab 6


# We want to focuses on positive benefit so I remove negative benefit, determined a threshold value for our standard benefit by considering the minimum and maximum of the 'benefit'_column.
# To analyze the factors contributing to higher versus lower benefits, we will split the data into two different datasets:
# 1. higher_benefit_  : _benefit_ $> 10000000$
# 2. lower_benefit_  : _benefit_ $\leq 10000000$
# 
# This division enable us to analyze the factors that contribute to higher versus lower benefits in each group.

# In[48]:


print(high_benefit.benefit.mean())


# In[49]:


ok = 10000000
notok = 0
high_benefit = df[df["benefit"] >= ok]
low_benefit  = df[df["benefit"] < ok  & (df["benefit"] > notok)]


# To determine how to achieve statistically better benefit on average and how The low_quality and high_quality distributions are different for budget , imdb, vote, review ,and facebook_likes, we perform hypothesis testing at the 0.1% significance level.
# Create histogram  and ECDF for both the high_benefit and low_benefit that show the distribution of each feature. The vertical lines in the histograms represent the mean of each distribution.By analyzing, we gain insights into how the chosen features are distributed differently between movies with high and low benefit ratings. The D statistic in the ECDF plots quantifies the maximum difference between the two distributions.

# In[50]:


fig, axes = plt.subplots(nrows = len(df.columns[6:10]), ncols = 3)

for idx, col in enumerate(df.columns[6:10]):
    qual_str = [fr"$benefit > {ok}$", fr"$benefit \leq {ok}$"]
    for idx2, df2 in enumerate([high_benefit ,low_benefit]):
        sns.histplot(data=df2,ax=axes[idx,idx2],x=col,line_kws={"lw":2},stat="density")
        mean = df2[col].mean()
        axes[idx,idx2].set_title(f"{col}, {qual_str[idx2]}: Mean = {mean:.4f}, n = {len(df2)}")
        axes[idx,idx2].set_xlabel(r"$x$")
    
    benefit_ECDF = ECDF(df[col])
    low_benefit_ECDF = ECDF(low_benefit[col])
    xax = np.linspace(min( low_benefit[col].min(),high_benefit[col].min()),max(high_benefit[col].max(),low_benefit[col].max()),1001)
    axes[idx,2].plot(xax,benefit_ECDF(xax),label=fr"$benefit > {ok}$")
    axes[idx,2].plot(xax,low_benefit_ECDF(xax),label=fr"$benrfit \leq {ok}$")
    axes[idx,2].legend(loc="best")
    D = np.max(np.abs(benefit_ECDF(xax) - low_benefit_ECDF(xax)))
    axes[idx,2].set_title(fr"{col}: $D$ = {D:.4f}")
    axes[idx,2].set_xlabel(r"$x$")
    axes[idx,2].set_ylabel(r"$\hat{F}_{X}(x)$")
    
fig.set_figwidth(fig.get_figwidth() * 3)
fig.set_figheight(fig.get_figheight() * len(df.columns[6:10])*1.25)


# In[51]:


fig, axes = plt.subplots(nrows = len(df.columns[20:26]), ncols = 3)

for idx, col in enumerate(df.columns[21:27]):
    qual_str = [fr"$benefit > {ok}$", fr"$benefit \leq {ok}$"]
    for idx2, df2 in enumerate([high_benefit ,low_benefit]):
        sns.histplot(data=df2,ax=axes[idx,idx2],x=col,line_kws={"lw":2},stat="density")
        mean = df2[col].mean()
        axes[idx,idx2].set_title(f"{col}, {qual_str[idx2]}: Mean = {mean:.4f}, n = {len(df2)}")
        axes[idx,idx2].set_xlabel(r"$x$")
    
    benefit_ECDF = ECDF(df[col])
    low_benefit_ECDF = ECDF(low_benefit[col])
    xax = np.linspace(min( low_benefit[col].min(),high_benefit[col].min()),max(high_benefit[col].max(),low_benefit[col].max()),1001)
    axes[idx,2].plot(xax,benefit_ECDF(xax),label=fr"$benefit > {ok}$")
    axes[idx,2].plot(xax,low_benefit_ECDF(xax),label=fr"$benrfit \leq {ok}$")
    axes[idx,2].legend(loc="best")
    D = np.max(np.abs(benefit_ECDF(xax) - low_benefit_ECDF(xax)))
    axes[idx,2].set_title(fr"{col}: $D$ = {D:.4f}")
    axes[idx,2].set_xlabel(r"$x$")
    axes[idx,2].set_ylabel(r"$\hat{F}_{X}(x)$")
    
fig.set_figwidth(fig.get_figwidth() * 3)
fig.set_figheight(fig.get_figheight() * len(df.columns[21:27])*1.25)


# # Kolmogorov-Smirnoff

# Conduct a Kolmogorov-Smirnoff_test to determine if any of the sample groups come from the same distributions.

# In[52]:


from scipy.stats import ks_2samp
alpha = 0.001

sig_cols = [] 

for idx, col in enumerate(top_rated.columns[6:10]):
    _,p_value_ks = ks_2samp(low_benefit[col],high_benefit[col])
    if p_value_ks < alpha:
        print(f"KS: The low and high quality distributions are significantly different for {col} at the {100*alpha}% significance level, p-value = {p_value_ks}")
        sig_cols.append(col)
    else:
        print(f"KS: The low and high quality distributions are not significantly different for {col} at the {100*alpha}% significance level, p-value = {p_value_ks}")


# In[66]:


from scipy.stats import ks_2samp
alpha = 0.001

sig_cols2 = [] 

for idx, col in enumerate(top_rated.columns[20:26]):
    _,p_value_ks = ks_2samp(low_benefit[col],high_benefit[col])
    if p_value_ks < alpha:
        print(f"KS: The low and high quality distributions are significantly different for {col} at the {100*alpha}% significance level, p-value = {p_value_ks}")
        sig_cols2.append(col)
    else:
        print(f"KS: The low and high quality distributions are not significantly different for {col} at the {100*alpha}% significance level, p-value = {p_value_ks}")


# # histograms and box-plots

# the feature that didn't pass the Kolmogorov-Smirnoff test, we generate histogram plots based on high_benefit and low_benefit, along with box plots.
# the histogram plots include vertical lines representing the means and medians. Additionally, the box plots append the skewness and kurtosis values to the titles.
# Upon reviewing these plots, we find out wich columns successfully pass a normality test.
# By visualizations, we gain insights into how the distribution of each feature differs between movies with high_benefits and low_benefits, providing a clearer understanding of the impact of each feature on movie benefit.

# In[54]:


#This observation suggests the potential necessity of employing a bootstrap method.
#Upon reviewing these plots and associated statistics 
#it becomes evident that the skewness and excess kurtosis values fall outside the range of -0.5 to 0.5. As a result
#it is unlikely that any of them would successfully pass a normality test.
#This observation suggests the potential necessity of employing a bootstrap method.
fig, axes = plt.subplots(nrows = len(sig_cols), ncols = 4, sharey="row")

for idx, col in enumerate(sig_cols):
    sns.histplot(data=high_benefit,ax=axes[idx,0],y=col,stat="density")
    sns.kdeplot(data=high_benefit,y=col,ax=axes[idx,0],color="k", linewidth=2)
    mean = high_benefit[col].mean()
    median = high_benefit[col].median()
    axes[idx,0].axhline(mean,ls="dashed",color="r")
    axes[idx,0].axhline(median,ls="dashed",color="g")
    high_benefit.plot(kind="box",y=col,ax=axes[idx,1],meanline=True,showmeans=True,meanprops={"color":"r"})
    skew = high_benefit[col].skew()
    kurtosis = high_benefit[col].kurtosis()
    
    axes[idx,0].set_ylabel(r"$x$")
    axes[idx,1].set_xticklabels([""])
    axes[idx,0].set_title(rf"{col}, $benefit > 7000000$: n = {len(high_benefit)}")
    axes[idx,1].set_title(rf"{col}, $benefit > 7000000$: Skew = {skew:.4f}, (E-)Kurtosis = {kurtosis:.4f}")
    
    sns.histplot(data=low_benefit,ax=axes[idx,2],y=col,stat="density")
    sns.kdeplot(data=low_benefit,y=col,ax=axes[idx,2],color="k", linewidth=2)
    mean = low_benefit[col].mean()
    median = low_benefit[col].median()
    axes[idx,2].axhline(mean,ls="dashed",color="r")
    axes[idx,2].axhline(median,ls="dashed",color="g")
    low_benefit.plot(kind="box",y=col,ax=axes[idx,3],meanline=True,showmeans=True,meanprops={"color":"r"})
    skew = low_benefit[col].skew()
    kurtosis = low_benefit[col].kurtosis()
    
    axes[idx,2].set_ylabel(r"$x$")
    axes[idx,3].set_xticklabels([""])
    axes[idx,2].set_title(rf"{col}, $benefit \leq 7000000$: n = {len(low_benefit)}")
    axes[idx,3].set_title(rf"{col}, $benefit \leq 7000000$: Skew = {skew:.4f}, (E-)Kurtosis = {kurtosis:.4f}")
    
fig.set_figwidth(fig.get_figwidth() * 4)
fig.set_figheight(fig.get_figheight() * len(sig_cols)*1.25)


# In[67]:


fig, axes = plt.subplots(nrows = len(sig_cols2), ncols = 4, sharey="row")

for idx, col in enumerate(sig_cols2):
    sns.histplot(data=high_benefit,ax=axes[idx,0],y=col,stat="density")
    sns.kdeplot(data=high_benefit,y=col,ax=axes[idx,0],color="k", linewidth=2)
    mean = high_benefit[col].mean()
    median = high_benefit[col].median()
    axes[idx,0].axhline(mean,ls="dashed",color="r")
    axes[idx,0].axhline(median,ls="dashed",color="g")
    high_benefit.plot(kind="box",y=col,ax=axes[idx,1],meanline=True,showmeans=True,meanprops={"color":"r"})
    skew = high_benefit[col].skew()
    kurtosis = high_benefit[col].kurtosis()
    
    axes[idx,0].set_ylabel(r"$x$")
    axes[idx,1].set_xticklabels([""])
    axes[idx,0].set_title(rf"{col}, $benefit > 7000000$: n = {len(high_benefit)}")
    axes[idx,1].set_title(rf"{col}, $benefit > 7000000$: Skew = {skew:.4f}, (E-)Kurtosis = {kurtosis:.4f}")
    
    sns.histplot(data=low_benefit,ax=axes[idx,2],y=col,stat="density")
    sns.kdeplot(data=low_benefit,y=col,ax=axes[idx,2],color="k", linewidth=2)
    mean = low_benefit[col].mean()
    median = low_benefit[col].median()
    axes[idx,2].axhline(mean,ls="dashed",color="r")
    axes[idx,2].axhline(median,ls="dashed",color="g")
    low_benefit.plot(kind="box",y=col,ax=axes[idx,3],meanline=True,showmeans=True,meanprops={"color":"r"})
    skew = low_benefit[col].skew()
    kurtosis = low_benefit[col].kurtosis()
    
    axes[idx,2].set_ylabel(r"$x$")
    axes[idx,3].set_xticklabels([""])
    axes[idx,2].set_title(rf"{col}, $benefit \leq 7000000$: n = {len(low_benefit)}")
    axes[idx,3].set_title(rf"{col}, $benefit \leq 7000000$: Skew = {skew:.4f}, (E-)Kurtosis = {kurtosis:.4f}")
    
fig.set_figwidth(fig.get_figwidth() * 4)
fig.set_figheight(fig.get_figheight() * len(sig_cols2)*1.25)


# # D'Agostino's_Ksquared_test and Lilliefors

# D'Agostino's K-squared test determine if either of the sample can be rejected as Gaussian for all features that failed the Kolmogorov-Smirnoff test. If both are rejected, we employ a Lilliefors test to ascertain their Gaussian nature.

# In[56]:


#as it is capable of handling both scenarios. The K-squared test follows a stricter rejection criterion
#relying on skewness and kurtosis, and given our sample size, we choose to utilize the Lilliefors test
# The results and the p-value for both tests will be printed.
#the K-squared test can only conclude if a sample is not Gaussian Therefore, we opt for the Lilliefors test.
from scipy.stats import normaltest
from statsmodels.stats.diagnostic import lilliefors

norm_cols = []
not_norm_cols = [] 

for col in sig_cols:
    _,p_value_n_q = normaltest(high_benefit[col])
    _,p_value_n_lq = normaltest(low_benefit[col])
    
    if (p_value_n_q < alpha) or (p_value_n_lq < alpha):
        print(f"KSquared: At least one {col} distribution is statistically not Gaussian at the {100*alpha}% significance level, ({p_value_n_q},{p_value_n_lq})")
        not_norm_cols.append(col)
    else:
        _,p_value_lf_q = lilliefors(high_benefit[col],dist="norm")
        _,p_value_lf_lq = lilliefors(low_benefit[col],dist="norm")
        
        if (p_value_lf_q < alpha) and (p_value_lf_lq < alpha):
            print(f"Lilliefors: Both {col} distributions are statistically Gaussian at the {100*alpha}% significance level, ({p_value_lf_q},{p_value_lf_lq})")
            norm_cols.append(col)
        else:
            print(f"Lilliefors: At least one {col} distributions are not statistically Gaussian at the {100*alpha}% significance level, ({p_value_lf_q},{p_value_lf_lq})")
            not_norm_cols.append(col)


# In[68]:


from scipy.stats import normaltest
from statsmodels.stats.diagnostic import lilliefors

norm_cols2 = []
not_norm_cols2 = [] 

for col in sig_cols2:
    _,p_value_n_q = normaltest(high_benefit[col])
    _,p_value_n_lq = normaltest(low_benefit[col])
    
    if (p_value_n_q < alpha) or (p_value_n_lq < alpha):
        print(f"KSquared: At least one {col} distribution is statistically not Gaussian at the {100*alpha}% significance level, ({p_value_n_q},{p_value_n_lq})")
        not_norm_cols2.append(col)
    else:
        _,p_value_lf_q = lilliefors(high_benefit[col],dist="norm")
        _,p_value_lf_lq = lilliefors(low_benefit[col],dist="norm")
        
        if (p_value_lf_q < alpha) and (p_value_lf_lq < alpha):
            print(f"Lilliefors: Both {col} distributions are statistically Gaussian at the {100*alpha}% significance level, ({p_value_lf_q},{p_value_lf_lq})")
            norm_cols.append2(col)
        else:
            print(f"Lilliefors: At least one {col} distributions are not statistically Gaussian at the {100*alpha}% significance level, ({p_value_lf_q},{p_value_lf_lq})")
            not_norm_cols2.append(col)


# In[58]:


def bootstrap_diff_means(data_1,data_2,num_bootstraps=10000,tail="two"):
    # Calculate means, variances, and sample sizes for the two datasets
    m_1 = np.mean(data_1)
    m_2 = np.mean(data_2)
    v_1 = np.var(data_1, ddof=1)
    v_2 = np.var(data_2, ddof=1)
    n_1 = len(data_1)
    n_2 = len(data_2)
    # Calculate the standard error and t-statistic for the difference of means
    se = np.sqrt(v_1/n_1 + v_2/n_2)
    t_stat = (m_1 - m_2)/se
    # Combine the datasets to calculate a common mean
    comb_m = (np.sum(data_1) + np.sum(data_2))/(n_1 + n_2)
    # Adjust the datasets by centering them around the common mean
    adj_col1 = data_1 - m_1 + comb_m
    adj_col2 = data_2 - m_2 + comb_m
    # Perform bootstrap resampling
    count = 0
    for _ in range(num_bootstraps):
        # Generate bootstrap samples for both datasets
        bs_1 = np.array([adj_col1[i] for i in np.random.randint(0,n_1,size=n_1)])
        bs_2 = np.array([adj_col2[i] for i in np.random.randint(0,n_2,size=n_2)]) 
         # Calculate means, variances, and standard errors for the bootstrap samples
        bs_m_1 = np.mean(bs_1)
        bs_m_2 = np.mean(bs_2)
        bs_v_1 = np.var(bs_1,ddof=1)
        bs_v_2 = np.var(bs_2,ddof=1)
        bs_se = np.sqrt(bs_v_1/n_1 + bs_v_2/n_2)
         # Calculate the bootstrap t-statistic
        bs_t_stat = (bs_m_1 - bs_m_2)/bs_se
        # Compare bootstrap t-statistic based on the chosen tail
        if tail == "two":
            if np.abs(bs_t_stat) >= np.abs(t_stat):
                count += 1
        elif tail == "less":
            if bs_t_stat <= t_stat:
                count += 1
        elif tail == "greater":
            if bs_t_stat >= t_stat:
                count += 1
                

    return m_1 - m_2, t_stat, (count+1)/(num_bootstraps+1)
    # Return the observed difference of means, observed t-statistic, and bootstrap p-value


# # Difference of two_means_test

# Bootstrapped difference of two means test to examine if the sample groups exhibit a distinct mean for all features that failed the D'Agostino's and Lilliefors tests.

# In[59]:


# this will take approximately 1minute
# The results for all features, including the p-values for both tests, will be printed.
sig_mean_cols = []
dbars = [] 
for col in not_norm_cols:
    
    data_1 = high_benefit[col].to_numpy()
    data_2 = low_benefit[col].to_numpy()
    
    if np.mean(data_1) - np.mean(data_2) < 0:
        tail = "less"
    else:
        tail = "greater"
    
    dbar, _, p_value_bs = bootstrap_diff_means(data_1,data_2,tail=tail)
    print(p_value_bs)
    if p_value_bs < alpha:
        print(f"Bootstrap: There is significant evidence to state that the average {col} levels for benefit > {ok} is different to benefit <= {ok} at the {100*alpha}% significance level, p-value = {p_value_bs}")
        sig_mean_cols.append(col)
        dbars.append(dbar)
    else:
        print(f"Bootstrap: There is not significant evidence to state that the average {col} levels for benefit > {ok} is different to benefit <= {ok} at the {100*alpha}% significance level, p-value = {p_value_bs}")


# In[69]:


# this will take approximately 1minute
sig_mean_cols2 = []
dbars2 = [] 
for col in not_norm_cols2:
    
    data_1 = high_benefit[col].to_numpy()
    data_2 = low_benefit[col].to_numpy()
    
    if np.mean(data_1) - np.mean(data_2) < 0:
        tail = "less"
    else:
        tail = "greater"
    
    dbar, _, p_value_bs = bootstrap_diff_means(data_1,data_2,tail=tail)
    print(p_value_bs)
    if p_value_bs < alpha:
        print(f"Bootstrap: There is significant evidence to state that the average {col} levels for benefit > {ok} is different to benefit <= {ok} at the {100*alpha}% significance level, p-value = {p_value_bs}")
        sig_mean_cols2.append(col)
        dbars2.append(dbar)
    else:
        print(f"Bootstrap: There is not significant evidence to state that the average {col} levels for benefit > {ok} is different to benefit <= {ok} at the {100*alpha}% significance level, p-value = {p_value_bs}")


# # Conclusion

# We conclude which features, on average, lead to statistically better benefit, and whether the levels need to be higher or lower to increase quality

# In[61]:


print(f"To get statistically better benefit (at the {100*alpha}% signficance level), on average we want")
for idx,(db,col) in enumerate(zip(dbars,sig_mean_cols)):
    if db < 0:
        print(f"{idx+1}: Lower levels of {col}")
    else:
        print(f"{idx+1}: Higher levels of {col}")


# In[70]:


print(f"To get statistically better benefit (at the {100*alpha}% signficance level), on average we want")
for idx,(db,col) in enumerate(zip(dbars2,sig_mean_cols2)):
    if db < 0:
        print(f"{idx+1}: Lower levels of {col}")
    else:
        print(f"{idx+1}: Higher levels of {col}")


# In[63]:


import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)


# # Hypothesis_Testing

# In[64]:


# I get the idea form lab 6


# 
# I sampled the movies 5044 times and split our sample into two groups, a high_benefit sample and a low_benefit sample   

# ## statement

# If I produce a film using my process and assign it a benefit rating, the average benefit will be higher when the average levels of actor_1_facebook_likes, actor_2_facebook_likes, actor_3_facebook_likes, and cast_total_facebook_likes are higher. This statement can be applied to all the columns we previously tested, but I have selected these specific columns as they can significantly contribute to the success of our project.

# ## Result

# The Kolmogorov-Smirnoff test at a 0.1% significance level indicates that actor_1_facebook_likes, actor_2_facebook_likes, actor_3_facebook_likes, and cast_total_facebook_likes for benefit ratings come from different distributions. D'Agostino's test shows that neither distribution is statistically Gaussian. Consequently, a Bootstrap test reveals that, on average, higher benefit ratings correspond to higher movie_facebook_likes levels. All tests reject the null hypotheses, concluding differences in distributions and averages between benefit ratings.

# In[65]:


## Mathematical_process
#\begin{align*}
#\mathbf{H}_0 \text{\it the sample groups come from the same distribution} \\
#\mathbf{H}_1 \text{\it the sample groups come from different distributions}
#\end{align*}
#I want to conduct a Kolmogorov-Smirnoff test on the two samples to check if they come from the same distribution 
#at the 0.1% significance level.
#Since the p-value from the Kolmogorov-Smirnoff test is less than our chosen significance(1.9042652737125398e-18),
#we reject the null hypothesis. That is, there is sufficient evidence to state that the distribution of 
#mdirector_facebook_likes for benefit ratings are different. 
#In order to check if both distributions are statistically Gaussian , I will conduct a two-pronged method:
#1.I will use the D'Agostino's test to decide whether either are not statistically Gaussian
#2.If both sample groups pass the D'Agostino's test, I will use Lilliefors test to conclude if both samples come from a Gassian 
#They both use the same mathematical formulation.
#Since the p-values for D'Agostino's test both came back less than our chosen significance(6.174829236042265e-248,0.0), 
#we reject the null hypothesis. This is sufficient evidence to state that the distribution of director_facebook_likes levels, for both benefit splits, have not come from a Gaussian distribution. 
#Since they both failed the D'Agostino's test, we don't need to use Lilliefors.
#Finally, I want to conduct a difference of means Bootstrap test on the two samples to check if the averages are different.
#In the hypothesis statement, we said "benefit rating will be higher on average if the levels of director_facebook_likes  
#are higher on average" -therefore we need to conduct a right-sided tail test: 
#$\mathbf{H}_0  :  \mu_{hq} - \mu_{lq} = 0$ 
#$\mathbf{H}_1 :  \mu_{hq} - \mu_{lq} > 0$
#$\mu_{hq}$ denotes the mean of the high_benefit and $\mu_{lq}$ denotes the mean of the low_benefit. 
#Since the p-value from the Bootstrapped test of the difference of the means is less than our chosen significance(9.999000099990002e-05), 
#we reject the null hypothesis. there is sufficient evidence to state that the average levels ofdirector_facebook_likes are higher if the benefit is higher on average.

