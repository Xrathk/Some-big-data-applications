########## Collaborative Filtering

# importing libraries
import numpy as np
import pandas as pd
import os
import sys
from scipy import spatial

# moving to correct directory
os.chdir(r'Desktop\ΜΕΤΑΠΤΥΧΙΑΚΟ\Μεταπτυχιακό ΕΚΠΑ Τηλεπικοινωνίες\\2ο Εξάμηνο\Διαχείρηση Μεγάλων Δεδομένων\Big Data Projects\Project 2\ml-25m')

chunk_size = 5000 # number of entries we import each time from .csv file

# converts combined movie ratings of a user to a usable list
def movieratings_to_list(movies):
    # stripping useless characters
    movies = movies.replace("[","")
    movies = movies.replace("]","")
    movies = movies.strip()
    movies = movies.split(",")
    # making a list of sublists: each sublist has 2 elements, a movieId and its rating
    for i in range(len(movies)):
        movies[i] = movies[i].replace("'","")
        movies[i] = movies[i].replace(" ","")
        movies[i] = movies[i].split(":")
        movies[i][1] = float(movies[i][1])
    return movies

# puts all the movies the user has watched in a set
def movies_user_watched(movies):
    movies_watched = set()
    for movie in movies:
        movies_watched.add(movie[0])
    return movies_watched

# calculating distance in preferences between the 2 users
# Calculating distance: if users have only one movie in common, euclidian distance is calculated. If they have > 1 movies in common, cosine distance is calculated
# in the similarity matrix, we save the distance between the user in question and the current user
# if the 2 users have no movies in common, distance is set to 10
# Arguments: ratings of user entered, ratings of current user, movies user entered has watched, movies current user has watched
def user_distance(r_user,r_currentuser,common):
    # creating lists with movies common movie ratings for both users so we can compare
    list_user = []
    list_current = []
    for movie in common:
        for mov in r_user:
            if mov[0] == movie:
                list_user.append(mov[1])
        for mov in r_currentuser:
            if mov[0] == movie:
                list_current.append(mov[1])
    
    # calculating distance
    if (len(common)==0):
        userdistance = 10
    elif (len(common)==1):
        userdistance = spatial.distance.euclidean(list_user, list_current)
    else:
        userdistance = spatial.distance.cosine(list_user, list_current)
    return userdistance

# Calculates whether the mean rating of a random movie is within the allowed interval from a user's favorite movies, for item-based collab. filtering
# interval is the interval defined by the favorite movies, rating is the rating
def in_interval(interval,rating):
    for i in range(5):
        if rating >= interval[i][0] and rating <= interval[i][1]:
            return True # if it's inthe interval, return true
    return False # if it isn't, return false

# finding jaccard similarity between genres/tags of 2 movies
def jaccard_distance(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return 1 - len(intersection) / len(union)

'''
### CREATING APPROPRIATE FILES FIRST (do it once)
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
#---------------------------------------------------------------------------------------------------------------------------------------------
'''

print("\nCollaborative Filtering:\n")

print("Make sure that you've created the files 'newRatings.csv' and 'newRatings_CF.csv'. Run FileCreation.py once to create them.\n")

# getting user's choice of collaborative filtering
print("\nWhat kind of recommendation do you want?")
print("1: User Based")
print("2: Item Based")
print("3: Combination of user based and item based")
choice = int(input("Enter the number (1, 2 or 3) that corresponds to your choice: "))
print("\n")

# checking choice validity
if choice != 1 and choice != 2 and choice != 3:
    sys.exit("Option not recognized. You must enter 1, 2 or 3 based on your choice. Exiting program...")

# printing appropriate message for choice 3
if choice == 3:
    print("You picked combined item-user based filtering. We will start with user-based filtering, then move on to item-based filtering.")
    print("The final recommendations will be a mix of the two.\n")

# getting user ID
user = input("Pick a user ID to recommend movies to: ")

# loading user's movie choices
found = 0 # user not found yet
for chunk in pd.read_csv('newRatings_CF.csv', chunksize=chunk_size):
    for  index, row in chunk.iterrows():
        if str(row['userId']) == user:
            user_movies = row['movie_ratings']
            found = 1
            break
    if found == 1:
        break

# exit program if user doesn't exist
if found == 1:
    print("User identified. Starting recommendation process...\n")
else:
    sys.exit("Invalid user ID. Exiting program...")

# putting user's preferences into a list
user_movies = movieratings_to_list(user_movies)
# creating a set of the movies the user has seen
movies_watched = movies_user_watched(user_movies)

chunk_num = 0 # current chunk of disk-based data reading

### Option 1: User-Based collaborative Filtering
if choice == 1 or choice == 3:
    user_similarity = [[0 for x in range(3)] for x in range(162541)] # user_similarity matrix for all 162541 users (contains only userID and distance from user in question)
    print("User-based filtering: Calculating Similarity matrix...")
    for chunk in pd.read_csv('newRatings_CF.csv', chunksize=chunk_size):
        # finding similarity between all users and current user
        for  index, row in chunk.iterrows():
            # adding userID to similarity list
            user_similarity[index][0] = index+1 # userId = index + 1
            # loading current user's preferences
            ratings = row['movie_ratings']
            ratings = movieratings_to_list(ratings)
            movies = movies_user_watched(ratings)
            # saving amount of common films with user
            common = movies_watched.intersection(movies)
            user_similarity[index][1] = len(common)
            # calculating distance in preferences between the 2 users
            distance = user_distance(user_movies,ratings,common)
            user_similarity[index][2] = distance # appending to matrix
            
        
        chunk_num = chunk_num + 1
        
    
        if chunk_num*chunk_size < 162541:
            print(chunk_num*chunk_size," out of 162541 users scanned...")
        else:
            print("Similarity matrix calculated!\n")
        
    # sorting list so most similar users go to the top
    user_similarity = sorted(user_similarity,key=lambda x: (x[2], -x[1]))

    '''
    # printing similarity list 
    for i in range(1000):
        print(user_similarity[i][0],"----------->",user_similarity[i][2],"---------->",user_similarity[i][1],"common films.")
        
    '''

    # printing users the user is most similar to 
    similarusers = []
    for i in range(11):
        if str(user_similarity[i][0])==user:
            continue
        similarusers.append(str(user_similarity[i][0]))
    similarusers_string = ",".join(similarusers)
    print("User with the ID:{} is most similar to the users with IDs:{}.".format(user,similarusers_string))
    print("\nPicking movie recommendations...\n")

    # finding the movies users similar to user in question have watched but he hasn't
    recommendations = 0 # we're looking for 20 movies
    movies_recommended = [] # empty list, we will fill it up with 20 movie Ids
    users_through = 0 # similar users we've scanned so far
    # iterating over dataframe again
    while recommendations < 21:
        new_recommends = 0 # new recommendations for new similar user
        found = 0 # similar user not found yet
        for chunk in pd.read_csv('newRatings_CF.csv', chunksize=chunk_size):
            for  index, row in chunk.iterrows():
                if str(row['userId'])==str(user_similarity[users_through][0]): # if ID is similar to a user in the similarity list  
                    print("Please wait...")
                    found = 1 # similar user found
                    # loading user's preferences      
                    ratings = row['movie_ratings']
                    ratings = movieratings_to_list(ratings)
                    movies = movies_user_watched(ratings)
                    # finding movies current user has watched and user in question hasn't
                    new_movies = movies - movies_watched
                    # adding these new movies to recommendation lists, if their rating is 4.0 or higher
                    for movie in new_movies:
                        # fetch movie rating
                        for i in range(len(ratings)):
                            if ratings[i][0] == movie:
                                rating = ratings[i][1]
                            # if rating is bigger than 4, add it to recommendations
                                if rating>=4.0:
                                    movies_recommended.append(movie)
                                    new_recommends = new_recommends + 1 
                        
                    recommendations = recommendations + new_recommends
                    users_through = users_through + 1 # on to next most similar user
                    break
            
            # user found, on to next user
            if found == 1:
                break
    
    # printing movie IDs to recommend
    print("\nRecommendation process complete!\n")
    
    # keeping only 20 recomemmendations
    while len(movies_recommended) > 20:
        movies_recommended.pop(-1) # deleting last element from list

    # converting movie Ids to movie names
    final_recommendations_user = dict.fromkeys(movies_recommended, 0) # dictionary with movieIds and movienames
    titles = 0 # how many titles we have found
    for chunk in pd.read_csv('movies.csv', chunksize=chunk_size):
        for  index, row in chunk.iterrows():
            if str(row['movieId']) in final_recommendations_user:
                final_recommendations_user[str(row['movieId'])] = row['title']
                titles = titles + 1
        if titles == 20:
            break
        
    if choice == 1:
        # printing final recommendations
        print("User with the ID:{} should watch the movies:".format(user))
        print("------------------------------------------------------------------------------")
        for movie_id, name in final_recommendations_user.items():
            print("{} corresponding to the ID:{}".format(name,movie_id))

        print("------------------------------------------------------------------------------\n")

### Option 2: Item-Based collaborative Filtering
if choice == 2 or choice == 3:
    print("Item-based filtering: Finding similar items to the ones the user has liked...\n")
    # first, we need to map the movies the user has rated by their reviews, and then find the max reviews
    user_movies = sorted(user_movies,key=lambda x: -x[1]) # sorting user's ratings
    
    # we will take the top 5 matrix and put them in a list
    movie_items = []
    for movie in user_movies:
        movie_items.append(movie[0])
        if len(movie_items)>4:
            break
    # we now have the user's top 5 movies, we must print them to the screen with the proper name
    # converting movie Ids to movie names
    movie_names = dict.fromkeys(movie_items, 0) # dictionary with movieIds and movienames
    movie_genres = dict.fromkeys(movie_items, 0) # dictionary with the genres so we can find similar items later
    titles = 0 # how many titles we have found
    for chunk in pd.read_csv('movies.csv', chunksize=chunk_size):
        for  index, row in chunk.iterrows():
            if str(row['movieId']) in movie_names:
                movie_names[str(row['movieId'])] = row['title']
                movie_genres[str(row['movieId'])] = row['genres']
                titles = titles + 1
        if titles == 5:
            break

    print("These are the favorite movies of the user with ID: {}".format(user))
    print("----------------------------------------------------------------------------")
    
    for movie_id, name in movie_names.items():
        print("{} corresponding to the ID: {}".format(name,movie_id))
    print("\n")

    # now we need to find the average ratings of all these movies, calculated by all the users
    movie_ratings = dict.fromkeys(movie_items, 0) # dictionary with ratings
    titles = 0 # how many movies we've found
    for chunk in pd.read_csv('newRatings.csv', chunksize=chunk_size):
        for  index, row in chunk.iterrows():
            if str(int(row['movieId'])) in movie_names:
                movie_ratings[str(int(row['movieId']))] = row['mean']
                titles = titles + 1
        if titles == 5:
            break
    
    # printing rating
    print("The above movies have the following ratings:")
    print("----------------------------------------------------------------------------")
    for movie_id, rating in movie_ratings.items():
        print("{} has a mean rating of: {}".format(movie_names[movie_id],rating))
    print("\nSearching for movies with ratings similar to the above...\n")
            
    # searching for similar items (aka items with similar ratings)
    # We will look for 1000 movies, all of which are within 0.05 points of what the mean reviews of any film
    # First we will find similar movies by rating, then by their genres (jaccard similarity)
    # the top 20 movies with the least jaccard distance will be recommended to the user

    # finding the appropriate interval a movie rating should be (within 0.05 of any of the 5 "favorite" movies)
    interval = [[0 for x in range(2)] for x in range(5)]
    i = 0
    for movie in movie_ratings:
        interval[i][0] = movie_ratings[movie]-0.05
        interval[i][1] = movie_ratings[movie]+0.05
        i = i+1

    print("Search complete. Now searching for movies with similar genres to the user's favorites...\n")
    # finding the similar movies (by rating). We're looking for initial_recommendations movies
    # we also want the movies with over 1000 reviews so user is recommended more known movies he can find
    initial_recommendations = 1000 # how many movies make the first draft (must be in the correct rating interval)
    movies_recommended = [] # empty list, we will fill it up with initial_recommendations movie Ids
    for chunk in pd.read_csv('newRatings.csv', chunksize=chunk_size):
        for  index, row in chunk.iterrows():
            if in_interval(interval,row['mean']) and row['count']>1000: # checking if movie rating is in the correct interval
                if str(int(row['movieId'])) not in movies_watched: # making sure the user hasn't watched this movie
                    movies_recommended.append(str(int(row['movieId'])))
            if len(movies_recommended) == initial_recommendations: # stop looking for new movies when you reach initial_recommendations
                break
        if len(movies_recommended) == initial_recommendations: # stop looking for new movies when you reach initial_recommendations
            break
    
    # now that we have movies similar in rating, we will also calculate the jaccard similarity to get even better recommendations
    # we will fetch the names and genres of the recommendations
    recommended_names = dict.fromkeys(movies_recommended, 0) # dictionary with movieIds and movienames
    recommended_genres = dict.fromkeys(movies_recommended, 0) # dictionary with the genres so we can find similar items later
    titles = 0 # how many titles we have found
    for chunk in pd.read_csv('movies.csv', chunksize=chunk_size):
        for  index, row in chunk.iterrows():
            if str(row['movieId']) in recommended_names:
                recommended_names[str(row['movieId'])] = row['title']
                recommended_genres[str(row['movieId'])] = row['genres']
                titles = titles + 1
        if titles == 500:
            break
    
    # now we must calculate jaccard similarity/distance of user's favorite movies and the above initial_recommendations
    # we will keep the jaccard distance that is the least from any of the 5 movies (not all 5)
    jaccard_distances = [[0 for x in range(5)] for x in range(initial_recommendations)] # 5 favorite movies, initial_recommendations 
    jaccard_distances_best = [[0 for x in range(2)] for x in range(initial_recommendations)] # keeping only movieID and the jaccard distance that's the least out of the 5 we're calculating for each recommendation
    i = 0 # this variable points to an index of the movies_recommended list so we can iterate through the dataset and fetch jaccard distances efficiently
    for chunk in pd.read_csv('movies.csv', chunksize=chunk_size):
        # calulating jaccard distance
        for  index, row in chunk.iterrows():
            genres = set(row['genres'].split("|"))
            if '(no genres listed)' in genres:
                continue # skip movies with no genres listed
            # check if recommended movie Id matches with current movie id
            if str(row['movieId']) in movies_recommended:
                j = 0 # for filling jaccard matrix
                for movie in movie_genres:
                    jaccard_distances[i][j] = jaccard_distance(genres,set(movie_genres[movie].split("|"))) # set operations 
                    j = j+1
                # finding minimum jaccard distance from any of the 5 movies
                min = 10
                for k in range(5):
                    if (jaccard_distances[i][k] <= min):
                        min = jaccard_distances[i][k]
                # keeping the least jaccard distance
                jaccard_distances_best[i][0] = str(row['movieId'])
                jaccard_distances_best[i][1] = min
                i = i+1

    # sorting the jaccard similarity matrix so we find top 20 movies with the least jaccard distance
    jaccard_distances_best = sorted(jaccard_distances_best,key=lambda x: x[1])
    # keeping only top 20
    while len(jaccard_distances_best) > 20:
        jaccard_distances_best.pop(-1)
    
    # adding top 20 to a dictionary
    final_recommendations_item = {}
    for i in range(20):
        final_recommendations_item[jaccard_distances_best[i][0]] = 0
    # getting movie names of the final recommendations...
    titles = 0 # how many titles we have found
    for chunk in pd.read_csv('movies.csv', chunksize=chunk_size):
        for  index, row in chunk.iterrows():
            if str(row['movieId']) in final_recommendations_item:
                final_recommendations_item[str(row['movieId'])] = row['title']
                titles = titles + 1
        if titles == 20:
            break

    print("Recommendation process complete!")
    if choice == 2:
        # We now have our 20 recommendations. Printing them is next...
        print("User with the ID:{} should watch the movies:".format(user))
        print("------------------------------------------------------------------------------")
        for movie_id, name in final_recommendations_item.items():
            print("{} corresponding to the ID:{}".format(name,movie_id))

        print("------------------------------------------------------------------------------\n")

### Option 3: Combined collaborative Filtering
if choice == 3:
    print("\nMaking a combined list with the movies recommended from user-based and item-based collaborative filtering...\n")

    # making a dictionary with 20 movies from the dictionaries final_recommendations_user (user based CF) and final_recommendations_item (item based CF)
    final_recommendations_combined = {} 
    k = 0 # we're taking 10 movies from each list
    for movie_id, name in final_recommendations_user.items(): # user based filtering
        if movie_id not in final_recommendations_combined: # make sure no movie is picked twice
            final_recommendations_combined[movie_id] = name
            k = k+1
            if k == 10: # take 10 elements from each list
                break

    k = 0 # we're taking 10 movies from each list 
    for movie_id, name in final_recommendations_item.items(): # item based filtering
        if movie_id not in final_recommendations_combined: # make sure no movie is picked twice
            final_recommendations_combined[movie_id] = name
            k = k+1
            if k == 10: # take 10 elements from each list
                break

    print("List complete!")
    # We now have our 20 recommendations. Printing them is next...
    print("User with the ID:{} should watch the movies:".format(user))
    print("------------------------------------------------------------------------------")
    for movie_id, name in final_recommendations_combined.items():
        print("{} corresponding to the ID:{}".format(name,movie_id))

    print("------------------------------------------------------------------------------\n")