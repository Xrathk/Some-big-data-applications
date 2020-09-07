### CREATING APPROPRIATE FILES FIRST (do it once)

import numpy as np
import pandas as pd
import os
import sys

print("\nCreating the appropriate files for K-Means clustering and collaborative filtering...\n")

# moving to correct directory
os.chdir(r'Desktop\ΜΕΤΑΠΤΥΧΙΑΚΟ\Μεταπτυχιακό ΕΚΠΑ Τηλεπικοινωνίες\\2ο Εξάμηνο\Διαχείρηση Μεγάλων Δεδομένων\Big Data Projects\Project 2\ml-25m')

# b.) Tags of movies 
#---------------------------------------------------------------------------------------------------------------------------------------------
# first, we must group the data so movies and their tags are grouped together
tags = pd.read_csv('tags.csv')
tags = tags.dropna() # drop null values
keep_col = ['movieId','tag']
new_tags = tags[keep_col]
new_tags = new_tags.groupby('movieId')['tag'].apply(set)
new_tags.to_csv("newTags.csv")          # saving new csv file
print("newTags.csv created!")
#---------------------------------------------------------------------------------------------------------------------------------------------

# c.) Rating data for movies (newRatings.csv)
#---------------------------------------------------------------------------------------------------------------------------------------------
# first, we must group the data new csv file contains a movieId and the average rating of that movie by all users
ratings = pd.read_csv('ratings.csv')
keep_col = ['movieId','rating']
new_ratings = ratings[keep_col] 

new_ratings = new_ratings.groupby('movieId')['rating'].agg(['mean', 'count']) # mean = mean rating of movie, count = amount of reviews for movie
new_ratings.to_csv("newRatings.csv")          # saving new csv file
print("newRatings.csv created!")
#---------------------------------------------------------------------------------------------------------------------------------------------

# d.) Merged movie data for all distance functions (mergedMovieData.csv)
#---------------------------------------------------------------------------------------------------------------------------------------------
# we have to combine the new tags file alongside the new ratings file, along with the movie genres file
movies = pd.read_csv('movies.csv')
ratings = pd.read_csv('newRatings.csv')
tags = pd.read_csv('newTags.csv')

# we will only use movies that have both tags and ratings
idList1 = list()  # list of valid movie IDs (tags)
for i in range(len(tags)):
    idList1.append(tags['movieId'][i])

idList2 = list()  # list of valid movie IDs (ratings)
for i in range(len(ratings)):
    idList2.append(ratings['movieId'][i])

idList = list(set(idList1).intersection(idList2)) # movies with both ratings and tags

# filtering non-valid movies from all 3 dataframes
movies =  movies[movies.movieId.isin(idList)]
ratings =  ratings[ratings.movieId.isin(idList)]
tags = tags[tags.movieId.isin(idList)]

# merging all 3 files together
merged = movies.merge(tags, on='movieId')
merged = merged.merge(ratings, on='movieId')

merged.to_csv("mergedMovieData.csv", index = False)          # saving new csv file
print("mergedMovieData.csv created!")
#---------------------------------------------------------------------------------------------------------------------------------------------

# Collaborative filtering: Creating appropriate file for user-user comparisons (newRatings_CF.csv)
#---------------------------------------------------------------------------------------------------------------------------------------------
# loading ratings
ratings = pd.read_csv('ratings.csv')
# deleting timestamps from ratings file so it's smaller and easier to use
ratings = ratings.drop(columns = ['timestamp'])


# creating new column with all movie-rating pairs for each user
ratings["movie_ratings"] = ratings['movieId'].astype(str) + ':' + ratings['rating'].astype(str)
ratings = ratings.drop(columns = ['movieId','rating']) 
new_ratings = ratings.groupby('userId')['movie_ratings'].apply(list)


new_ratings.to_csv("newRatings_CF.csv", index = True, index_label = 'userId')          # saving new csv file (newRatings_CF for collaborative filtering)
print("newRatings_CF.csv created!")
#---------------------------------------------------------------------------------------------------------------------------------------------

print("\nAll required files created. \n")