########## K-means clustering 

import numpy as np
import pandas as pd
import os
from collections import Counter 
from ast import literal_eval
from numpy import dot
from numpy.linalg import norm
import sys
import time

# finding jaccard similarity between genres/tags of 2 movies
def jaccard_distance(set1, set2):
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return 1 - len(intersection) / len(union)


# finding cosine similarity between ratings of 2 movies
def cosine_distance(movie1, movie2):
    cos_sim = dot(movie1, movie2)/(norm(movie1)*norm(movie2))
    return 1-cos_sim
    

# moving to correct directory
os.chdir(r'Desktop\ΜΕΤΑΠΤΥΧΙΑΚΟ\Μεταπτυχιακό ΕΚΠΑ Τηλεπικοινωνίες\\2ο Εξάμηνο\Διαχείρηση Μεγάλων Δεδομένων\Big Data Projects\Project 2\ml-25m')

chunk_size = 9000 # number of entries we import each time from .csv file

print("\nK-Means Clustering:\n")

print("Make sure that you've created the files 'newTags.csv', 'newRatings.csv' and 'mergedMovieData.csv'. Run FileCreation.py once to create them.\n")

# getting user's choice of collaborative filtering
print("\nWhat kind of K-means clustering do you want?")
print("1: d1 = Jaccard similarity based on the genres of movies")
print("2: d2 = Jaccard similarity based on the tags of movies")
print("3: d3 = Cosine similarity based on the ratings of the movies")
print("4: d4 = 0.3*d1 + 0.25*d2 + 0.45*d3")
choice = int(input("Enter the number (1, 2, 3 or 4) that corresponds to your choice: "))
print("\n")

# checking validity of choice
if choice != 1 and choice != 2 and choice != 3 and choice != 4:
    sys.exit("Option not recognized. You must enter 1, 2, 3 or 4 based on your choice. Exiting program...")

# number of clustroids (2-10) 
K = int(input("How many clusters do you want to use (enter a number from 2 to 10)? "))

# checking validity of user input
allowed = [2,3,4,5,6,7,8,9,10]
if K not in allowed:
    sys.exit("Invalid number of clusters. This program supports K-Means clustering of 2 to 10 clusters. Exiting program...")

print("Starting K-Means clustering with {} clusters...\n".format(K))

# a) d1 = jaccard similarity of genres of movies
if choice == 1:
    # initial cluster info
    initial_clustroids = [0,3,5,36,162,186,187,253,263,400]
    initialization = False # initial clustroids not yet picked
    clustroids = [set() for x in range(K)]
    cluster_total = [0 for x in range(K)]

    chunk_num = 1 # current data chunk (because we have a disk-based implementation)

    # iterating through dataset, once chunk at a time
    for chunk in pd.read_csv('movies.csv', chunksize=chunk_size):
    
        # initializing jaccard distance array 
        jaccard_distances = [[0 for x in range(K)] for x in range(chunk_size)] 
        clusters_points = [0 for x in range(chunk_size)] # which cluster the movie belongs to

        # initializing dictionary with genres to calculate new clustroids
        dict_cluster = [dict() for x in range(K)]

        # initializing current chunk cluster data
        cluster_total_currentchunk = [0 for x in range(K)]

        # picking initial clustroids 
        if (initialization == False):
            iter = 0
            for point in initial_clustroids:
                genres = set(chunk.iloc[point]['genres'].split("|"))
                clustroids[iter] = genres
                iter = iter + 1
                if iter >= K: 
                    break
            
            # printing clustroids
            print("Initial clusters are:")
            index = 1
            for clustroid in clustroids:
                print("Clustroid " + str(index) +":",clustroid)
                index = index + 1

            initialization = True # initialization complete
            time.sleep(4) # enough time to see initial clusters
        
        for  index, row in chunk.iterrows():
            index = index%chunk_size
            genres = set(row['genres'].split("|"))
            if '(no genres listed)' in genres:
                continue # skip movies with no genres listed
            # calculating jaccard distances from clustroids
            for i in range(K):
                jaccard_distances[index][i] = jaccard_distance(genres,clustroids[i])
            # finding which cluster the movie belongs to
            min = 1
            index_min = 1
            for i in range(K):
                if (jaccard_distances[index][i] <= min):
                    min = jaccard_distances[index][i]
                    index_min = i

            clusters_points[index] = index_min + 1
            # adding up genres so we can find the new clustroids later 
            for genre in genres:
                if genre in dict_cluster[index_min].keys():
                    dict_cluster[index_min][genre] = dict_cluster[index_min][genre] + 1
                else:
                    dict_cluster[index_min][genre] = 1

        '''
        # printing the cluster each movie belongs too
        for  index, row in chunk.iterrows():
            print("Movie",row['title'],"belongs to cluster " + str(clusters_points[index]) + ".")
        '''

        # calculating how many points in each cluster
        for i in range(chunk.shape[0]):
            cluster_total_currentchunk[clusters_points[i]-1] = cluster_total_currentchunk[clusters_points[i]-1] + 1 # current chunk
            cluster_total[clusters_points[i]-1] = cluster_total[clusters_points[i]-1] + 1 # in total

        # printing info about current chunk:
        print("\nInfo after chunk " + str(chunk_num) + ".")

        # printing how many points belong in each cluster
        print("")
        for i in range(K):
            print("Cluster " + str(i+1) + ":", cluster_total[i])

        # changing clustroids
        print("")
        # printing genres in clusters and how many times we find them
        for i in range(K):
            print("Cluster " + str(i+1) + ":")
            for x in dict_cluster[i].keys():
                print(x,"=>",dict_cluster[i][x])
            print("")

        # creating new clusters (picking top 3 genres for each cluster)
        for i in range(K):
            clustroids[i].clear()
            k = Counter(dict_cluster[i]) 
            # Finding 3 highest values 
            high = k.most_common(3)  
            for x in high:
                clustroids[i].add(x[0])
            # printing new clustroid
            print("New clustroid " + str(i+1) + " :",clustroids[i])

        print("")
        chunk_num = chunk_num + 1

    # printing final clustroids
    print("Final clusters are:")
    index = 1
    for clustroid in clustroids:
        if len(clustroid) == 0:
            continue
        print("Clustroid " + str(index) +":",clustroid)
        index = index + 1

    print("")



# b) d2 = jaccard similarity of tags of movies
if choice == 2:
    # initial cluster info
    initial_clustroids = [0,9,15,47,67,104,116,141,204,248]
    initialization = False # initial clustroids not yet picked
    clustroids = [set() for x in range(K)]
    cluster_total = [0 for x in range(K)]

    chunk_num = 1 # current data chunk (because we have a disk-based implementation)
    
    for chunk in pd.read_csv('newTags.csv', chunksize=chunk_size):
        # initializing jaccard distance array 

        jaccard_distances = [[0 for x in range(K)] for x in range(chunk_size)] 
        clusters_points = [0 for x in range(chunk_size)]

        # initializing dictionary with tags to calculate new clustroids
        dict_cluster = [dict() for x in range(K)]

        # initializing current chunk cluster data
        cluster_total_currentchunk = [0 for x in range(K)]

        # picking initial clustroids 
        if (initialization == False):
            iter = 0
            for point in initial_clustroids:
                tags = eval(chunk.iloc[point]['tag'])
                clustroids[iter] = tags
                iter = iter + 1
                if iter >= K: 
                    break

            # printing clustroids
            print("Initial clusters are:")
            index = 1
            for clustroid in clustroids:
                print("Clustroid " + str(index) +":",clustroid)
                index = index + 1

            initialization = True # initialization complete
            time.sleep(4) # enough time to see initial clusters
        
        for  index, row in chunk.iterrows():
            index = index%chunk_size
            tags = literal_eval(row['tag'])
            
            # calculating jaccard distances from clustroids
            for i in range(K):
                jaccard_distances[index][i] = jaccard_distance(tags,clustroids[i])
            # finding which cluster the movie belongs to
            min = 1
            index_min = 1
            for i in range(K):
                if (jaccard_distances[index][i] <= min):
                    min = jaccard_distances[index][i]
                    index_min = i

            clusters_points[index] = index_min + 1
            # adding up tags so we can find the new clustroids later 
            for tag in tags:
                if tag in dict_cluster[index_min].keys():
                    dict_cluster[index_min][tag] = dict_cluster[index_min][tag] + 1
                else:
                    dict_cluster[index_min][tag] = 1

        # calculating how many points in each cluster
        for i in range(chunk.shape[0]):
            cluster_total_currentchunk[clusters_points[i]-1] = cluster_total_currentchunk[clusters_points[i]-1] + 1 # current chunk
            cluster_total[clusters_points[i]-1] = cluster_total[clusters_points[i]-1] + 1 # in total

        # printing info about current chunk:
        print("\nInfo after chunk " + str(chunk_num) + ".")

        # printing how many points belong in each cluster
        print("")
        for i in range(K):
            print("Cluster " + str(i+1) + ":", cluster_total[i])

        
        print("")
        # creating new clusters (picking top 10 tags for each cluster)
        for i in range(K):
            clustroids[i].clear()
            k = Counter(dict_cluster[i]) 
            # Finding 10 highest values 
            high = k.most_common(10)  
            for x in high:
                clustroids[i].add(x[0])
            # printing new clustroid
            print("New clustroid " + str(i+1) + " :",clustroids[i])

        print("")
        chunk_num = chunk_num + 1

    # printing final clustroids
    print("Final clusters are:")
    index = 1
    for clustroid in clustroids:
        if len(clustroid) == 0:
            continue
        print("Clustroid " + str(index) +":",clustroid)
        index = index + 1

    print("")


# c) d3 = cosine similarity of movie ratings
if choice == 3:
    # initial cluster info
    initial_clustroids = [0,1,3,32,179,211,215,108,250,257]
    initialization = False # initial clustroids not yet picked
    clustroids = [0 for x in range(K)]
    cluster_total = [0 for x in range(K)]

    # initializing list with ratings to calculate new clustroids
    list_cluster = [[0,0] for x in range(K)]

    chunk_num = 1 # current data chunk (because we have a disk-based implementation)

    for chunk in pd.read_csv('newRatings.csv', chunksize=chunk_size):
        # initializing cosine distance array 

        cosine_distances = [[0 for x in range(K)] for x in range(chunk_size)] 
        clusters_points = [0 for x in range(chunk_size)]


        # initializing current chunk cluster data
        cluster_total_currentchunk = [0 for x in range(K)]

        # picking initial clustroids 
        if (initialization == False):
            iter = 0
            for point in initial_clustroids:
                rating = [chunk.iloc[point]['mean'],chunk.iloc[point]['count']]
                clustroids[iter] = rating
                iter = iter + 1
                if iter >= K: 
                    break

            # printing clustroids
            print("Initial clusters are:")
            index = 1
            for clustroid in clustroids:
                print("Clustroid " + str(index) +":",clustroid)
                index = index + 1

            initialization = True # initialization complete
            
        
        
        for  index, row in chunk.iterrows():
            index = index%chunk_size
            rating = [chunk.iloc[index]['mean'],chunk.iloc[index]['count']]
            # calculating cosine distances from clustroids
            for i in range(K):
                cosine_distances[index][i] = cosine_distance(rating,clustroids[i])

            # finding which cluster the movie belongs to
            min = 1
            index_min = 1
            for i in range(K):
                if (cosine_distances[index][i] <= min):
                    min = cosine_distances[index][i]
                    index_min = i

            clusters_points[index] = index_min + 1
            
            
            # finding new means so we can find the new clustroids later 
            for k in range(2):
                if list_cluster[index_min][k] != 0:
                    if k == 0:
                        list_cluster[index_min][k] = list_cluster[index_min][k] + chunk.iloc[index]['mean']
                    else:
                        list_cluster[index_min][k] = list_cluster[index_min][k] + chunk.iloc[index]['count']
                else:
                    if k == 0:
                        list_cluster[index_min][k] = chunk.iloc[index]['mean']
                    else:
                        list_cluster[index_min][k] = chunk.iloc[index]['count']
            
        
        # calculating how many points in each cluster
        for i in range(chunk.shape[0]):
            cluster_total_currentchunk[clusters_points[i]-1] = cluster_total_currentchunk[clusters_points[i]-1] + 1 # current chunk
            cluster_total[clusters_points[i]-1] = cluster_total[clusters_points[i]-1] + 1 # in total
            

        # printing info about current chunk:
        print("\nInfo after chunk " + str(chunk_num) + ".")

        # printing how many points belong in each cluster
        print("")
        for i in range(K):
            print("Cluster " + str(i+1) + ":", cluster_total[i])
        
    
        print("")
        # creating new clusters (averaging mean and count)
        for i in range(K):
            clustroids[i].clear()
            #calculating new clustroid
            clustroids[i].append(list_cluster[i][0]/cluster_total[i]) # new average rating
            clustroids[i].append(list_cluster[i][1]/cluster_total[i]) # new count
            # printing new clustroid
            print("New clustroid " + str(i+1) + " :",clustroids[i])
        
        print("")
        chunk_num = chunk_num + 1


    # printing final clustroids
    print("Final clusters are:")
    index = 1
    for clustroid in clustroids:
        if len(clustroid) == 0:
            continue
        print("Clustroid " + str(index) +":",clustroid)
        index = index + 1

    print("")




# d) d4 = 0.3*d1 + 0.25*d2 + 0.45*d3
if choice == 4:
    # initial cluster info
    initial_clustroids = [0,1,3,5,9,15,32,36,67,162]
    initialization = False # initial clustroids not yet picked
    clustroids = [set() for x in range(K)]
    cluster_total = [0 for x in range(K)]

    # initializing list with ratings to calculate new clustroids
    list_cluster = [[0,0] for x in range(K)]

    chunk_num = 1 # current data chunk (because we have a disk-based implementation)

    for chunk in pd.read_csv('mergedMovieData.csv', chunksize=chunk_size):
        # initializing distance arrays

        d1 = [[0 for x in range(K)] for x in range(chunk_size)]  # genres (jaccard)
        d2 = [[0 for x in range(K)] for x in range(chunk_size)]  # tags (jaccard)
        d3 = [[0 for x in range(K)] for x in range(chunk_size)]  # ratings (cosine)
        d_total = [[0 for x in range(K)] for x in range(chunk_size)]  # combined distances
        clusters_points = [0 for x in range(chunk_size)]

        # initializing dictionary with genres to calculate new clustroids
        dict_cluster_genre = [dict() for x in range(K)]

        # initializing dictionary with tags to calculate new clustroids
        dict_cluster_tags = [dict() for x in range(K)]

        # initializing current chunk cluster data
        cluster_total_currentchunk = [0 for x in range(K)]

        # picking initial clustroids 
        if (initialization == False):
            iter = 0
            for point in initial_clustroids:
                moviedata = [set(chunk.iloc[point]['genres'].split("|")), eval(chunk.iloc[point]['tag']),[chunk.iloc[point]['mean'],chunk.iloc[point]['count']]]
                #moviedata = set(chunk.iloc[point]['genres'].split("|"))
                clustroids[iter] = moviedata
                iter = iter + 1
                if iter >= K: 
                    break

            # printing clustroids
            print("Initial clusters are:")
            index = 1
            for clustroid in clustroids:
                print("Clustroid " + str(index) +":",clustroid)
                index = index + 1

            initialization = True # initialization complete
            time.sleep(4) # enough time to see initial clusters
        
        
        for  index, row in chunk.iterrows():
            index = index%chunk_size
            moviedata = [set(chunk.iloc[index]['genres'].split("|")), eval(chunk.iloc[index]['tag']),[chunk.iloc[index]['mean'],chunk.iloc[index]['count']]]

            # calculating jaccard distances from clustroids (genres)
            for i in range(K):
                d1[index][i] = jaccard_distance(moviedata[0],clustroids[i][0])

            # calculating jaccard distances from clustroids (tags)
            for i in range(K):
                d2[index][i] = jaccard_distance(moviedata[1],clustroids[i][1])

            # calculating cosine distances from clustroids
            for i in range(K):
                d3[index][i] = cosine_distance(moviedata[2],clustroids[i][2])

            # calculating combined distance array
            for i in range(K):
                d_total[index][i] = 0.3*d1[index][i] + 0.25*d2[index][i] + 0.45*d3[index][i]

            # finding which cluster the movie belongs to
            min = 1
            index_min = 1
            for i in range(K):
                if (d_total[index][i] <= min):
                    min = d_total[index][i]
                    index_min = i

            clusters_points[index] = index_min + 1

            # adding up genres so we can find the new clustroids later
            for genre in moviedata[0]:
                if genre in dict_cluster_genre[index_min].keys():
                    dict_cluster_genre[index_min][genre] = dict_cluster_genre[index_min][genre] + 1
                else:
                    dict_cluster_genre[index_min][genre] = 1

            # adding up tags so we can find the new clustroids later 
            for tag in moviedata[1]:
                if tag in dict_cluster_tags[index_min].keys():
                    dict_cluster_tags[index_min][tag] = dict_cluster_tags[index_min][tag] + 1
                else:
                    dict_cluster_tags[index_min][tag] = 1
            
            # finding new means so we can find the new clustroids later 
            for k in range(2):
                if list_cluster[index_min][k] != 0:
                    if k == 0:
                        list_cluster[index_min][k] = list_cluster[index_min][k] + moviedata[2][0]
                    else:
                        list_cluster[index_min][k] = list_cluster[index_min][k] + moviedata[2][1]
                else:
                    if k == 0:
                        list_cluster[index_min][k] = moviedata[2][0]
                    else:
                        list_cluster[index_min][k] = moviedata[2][1]
            
        
        # calculating how many points in each cluster
        for i in range(chunk.shape[0]):
            cluster_total_currentchunk[clusters_points[i]-1] = cluster_total_currentchunk[clusters_points[i]-1] + 1 # current chunk
            cluster_total[clusters_points[i]-1] = cluster_total[clusters_points[i]-1] + 1 # in total
            

        # printing info about current chunk:
        print("\nInfo after chunk " + str(chunk_num) + ".")

        # printing how many points belong in each cluster
        print("")
        for i in range(K):
            print("Cluster " + str(i+1) + ":", cluster_total[i])
        
        
        ## CREATING NEW CLUSTERS
        print("")
        # creating new clusters (picking top 3 genres for each cluster)
        for i in range(K):
            clustroids[i][0].clear()
            k = Counter(dict_cluster_genre[i]) 
            # Finding 3 highest values 
            high = k.most_common(3)  
            for x in high:
                clustroids[i][0].add(x[0])
            

        # creating new clusters (picking top 10 tags for each cluster)
        for i in range(K):
            clustroids[i][1].clear()
            k = Counter(dict_cluster_tags[i]) 
            # Finding 10 highest values 
            high = k.most_common(10)  
            for x in high:
                clustroids[i][1].add(x[0])
            

        # creating new clusters (averaging mean and count)
        for i in range(K):
            clustroids[i][2].clear()
            #calculating new clustroid
            clustroids[i][2].append(list_cluster[i][0]/cluster_total[i]) # new average rating
            clustroids[i][2].append(list_cluster[i][1]/cluster_total[i]) # new count
        

        # printing NEW clustroids
        print("")
        index = 1
        for clustroid in clustroids:
            print("Clustroid " + str(index) +":",clustroid)
            index = index + 1

        print("")
        chunk_num = chunk_num + 1
        
    # printing final clustroids
    print("Final clusters are:")
    index = 1
    for clustroid in clustroids:
        if len(clustroid) == 0:
            continue
        print("Clustroid " + str(index) +":",clustroid)
        index = index + 1

    print("")
        