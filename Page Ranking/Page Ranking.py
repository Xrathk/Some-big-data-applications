########## Page Ranking ##########

# importing libraries
import collections
import random
import os
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# moving to correct directory
os.chdir(r'Desktop\ΜΕΤΑΠΤΥΧΙΑΚΟ\Μεταπτυχιακό ΕΚΠΑ Τηλεπικοινωνίες\\2ο Εξάμηνο\Διαχείρηση Μεγάλων Δεδομένων\Big Data Projects\Project 3')
start_time = time.time() # timekeeping

# Page-Ranking function (Erotima A)
# Arguments are a dictionary of the pages and the pages they point to, a dictionary of the pages and how many pages, they point to, and the iterations of the algorithm)
def PageRank(nodes_edges,edges_total,iterations):
    # initializing at one
    rankings = dict() # empty list - column one is ID, column two is rank
    for key in edges_total.keys(): # total amount of nodes
        rankings[key]=1

    new_rankings = dict() # new rankings after an iteration go here. rankings will be updated when iteration is over
    for key in edges_total.keys(): # total amount of nodes
        new_rankings[key]=1


    # starting iterations
    for i in range(iterations):
        # difference ( converge criteria )
        difference = 0

        for pageID in rankings.keys(): 
            new_value = 0 # initializing sum
            if pageID in nodes_edges:
                for page in nodes_edges[pageID]: # finding new ranking of a page
                    new_value = new_value + (1/edges_total[page])*rankings[page]
                    #print(pageID,page,new_value)
                    
           
            new_rankings[pageID] = new_value # finding new value
            
        print("Iteration {} complete!".format(i+1))   
        for p in rankings.keys(): # taking new rankings after ith iteration
            difference = difference + abs(new_rankings[p]-rankings[p]) # calculating difference of new value and old value. If it is less than 0.01, we have reached convergence
            rankings[p] = new_rankings[p]

        # checking convergence
        print("Difference is:",difference)
        
        if difference<1: # convergence achieved if total sum of difference less than 1 (aka each new value changed less than 1/N on average)
            print("Convergence achieved!")
            break
        
    
    print("\nPage ranking algorithm for {} iterations complete! Here are the final results...".format(iterations))
    # ordered dictionary, pages ranked by how many pages they point to (printing top 20)
    ordered_rankings = collections.OrderedDict(sorted(rankings.items(), key=lambda t: -t[1]))
    i = 1
    print("\nTop 20 pages:")
    print("-----------------------------------------------------------------------------------------------------")
    for key,value in ordered_rankings.items():
        print("Page {} has rank: {}".format(key,value))
        i = i+1
        if i == 20:
            break

    # ordered dictionary, pages ranked by how many pages they point to (printing bottom 20)
    ordered_rankings = collections.OrderedDict(sorted(rankings.items(), key=lambda t: t[1]))
    i = 1
    print("\nBottom 20 pages:")
    print("-----------------------------------------------------------------------------------------------------")
    for key,value in ordered_rankings.items():
        print("Page {} has rank: {}".format(key,value))
        i = i+1
        if i == 20:
            break
    
    print("")
    # plotting histograms
    # keeping only page ranks in  the list and not the IDs
    ranks = []
    for key, value in rankings.items():
        ranks.append(float(value))

    # plotting
    ranks = pd.Series(ranks)

    plt.figure(figsize=[10,8])
    plt.hist(x=ranks, bins=150, color='r',alpha=0.7, rwidth=0.85)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank',fontsize=15)
    plt.ylabel('Number of pages',fontsize=15)
    plt.title('Page Ranking (simple version)')
    plt.show()


# Page-Ranking function (Erotima B,C,D)
# Arguments are a dictionary of the pages and the pages they point to, a dictionary of the pages and how many pages, they point to, and the iterations of the algorithm)
def ImprovedPageRank(nodes_edges,edges_total,iterations,a_factor):
    # initializing at one
    rankings = dict() # empty list - column one is ID, column two is rank
    for key in edges_total.keys(): # total amount of nodes
        rankings[key]=1

    new_rankings = dict() # new rankings after an iteration go here. rankings will be updated when iteration is over
    for key in edges_total.keys(): # total amount of nodes
        new_rankings[key]=1


    # starting iterations
    for i in range(iterations):
        difference = 0 # initializing convergence criteria
        for pageID in rankings.keys(): 
            # added factor ((1-a)*In)
            added_factor = (1 - a_factor)*1
            new_value = 0 # initializing sum
            if pageID in nodes_edges:
                for page in nodes_edges[pageID]: # finding new ranking of a page
                    new_value = new_value + (1/edges_total[page])*rankings[page]
                    
                    
            new_value = a_factor*new_value
            new_rankings[pageID] = new_value + added_factor # finding new value
            
        print("Iteration {} complete!".format(i+1))   
        for p in rankings.keys(): # taking new rankings after ith iteration
            difference = difference + abs(new_rankings[p]-rankings[p]) # calculating difference of new value and old value. If it is less than 0.01, we have reached convergence
            rankings[p] = new_rankings[p]

        # checking convergence
        print("Difference is:",difference)
        
        if difference<1: # convergence achieved if total sum of difference less than 1 (aka each new value changed less than 1/N on average)
            print("Convergence achieved!")
            break
        
    
    
    print("\nPage ranking algorithm for {} iterations complete! Here are the final results...".format(iterations))
    # ordered dictionary, pages ranked by how many pages they point to (printing top 20)
    ordered_rankings = collections.OrderedDict(sorted(rankings.items(), key=lambda t: -t[1]))
    i = 1
    print("\nTop 20 pages:")
    print("-----------------------------------------------------------------------------------------------------")
    for key,value in ordered_rankings.items():
        print("Page {} has rank: {}".format(key,value))
        i = i+1
        if i == 20:
            break

    # ordered dictionary, pages ranked by how many pages they point to (printing bottom 20)
    ordered_rankings = collections.OrderedDict(sorted(rankings.items(), key=lambda t: t[1]))
    i = 1
    print("\nBottom 20 pages:")
    print("-----------------------------------------------------------------------------------------------------")
    for key,value in ordered_rankings.items():
        print("Page {} has rank: {}".format(key,value))
        i = i+1
        if i == 20:
            break
    
    print("")

    # plotting histograms
    # keeping only page ranks in  the list and not the IDs
    ranks = []
    for key, value in rankings.items():
        ranks.append(float(value))

    # plotting
    ranks = pd.Series(ranks)

    plt.figure(figsize=[10,8])
    plt.hist(x=ranks, bins=150, color='r',alpha=0.7, rwidth=0.85)
    
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Rank',fontsize=15)
    plt.ylabel('Number of pages',fontsize=15)
    title = 'Page Ranking for ' + 'a = ' + str(a_factor)
    plt.title(title)
    plt.show()
    

print("\nPage Ranking:\n")

### Reading Data
# instead of matrix, creating a dictionary with a page ID as key and a list of all its edges as the value, to save space
nodes_edges = dict()
# dictionary of a page ID and how many pages it is pointed by
edges_total = dict()
# set with all possible pages 
allPages = set()

# reading web-google.txt and storing all links in an appropriate form
filename = 'web-google.txt'
lines = 0 # counting lines 

print("Loading data...")

# open file
with open(filename) as f:
    for line in f:
        lines=lines+1
        if lines>4:
            new_line=line.split("\t")
            new_line[1] = new_line[1][:-1]
            pageID = new_line[0] # pageID
            edge = new_line[1] # edge

            # keeping all possible pages
            allPages.add(pageID)
            allPages.add(edge)

            # initializing, if seeing page ID for first time
            if edge not in nodes_edges:
                nodes_edges[edge] = []
                nodes_edges[edge].append(pageID)
            else: # if already initialized, add new edge to dictionary
                nodes_edges[edge].append(pageID)
              
            # initializing, if page hasn't been pointed at before
            if pageID not in edges_total:
                edges_total[pageID] = 1
            else: # if already pointed, increase by one
                edges_total[pageID] = edges_total[pageID] + 1
                
print("Loading complete!")

# handling dangling nodes: each page pointing to nowhere now points to a random page
allPages = list(allPages) # to pick random page to point to
for page in allPages:
    if page not in edges_total:
        edges_total[page] = 1
        random_choice = random.choice(allPages) # pointing to random page
        if random_choice not in nodes_edges:
            nodes_edges[random_choice] = []
            nodes_edges[random_choice].append(page)
        else:
            nodes_edges[random_choice].append(page)
        

### Page Ranking Algorithm
print("\nStarting iterations...")

### Page Ranking
# defining iterations

#iterations = 10
#iterations = 50
#iterations = 100
iterations = 200

# PageRanking function

'''
# Erotima A ) Simple Version of the Page Ranking algorithm
PageRank(nodes_edges,edges_total,iterations)
'''
# Erotima B ) Improved of the Page Ranking algorithm
# defining A factor

# 1st case
a_factor = 0.2
ImprovedPageRank(nodes_edges,edges_total,iterations,a_factor)

'''
# 2nd case
a_factor = 0.85
ImprovedPageRank(nodes_edges,edges_total,iterations,a_factor)
'''

end_time = time.time() # execution finished
print("Total time elapsed: {} seconds.".format(end_time-start_time))

