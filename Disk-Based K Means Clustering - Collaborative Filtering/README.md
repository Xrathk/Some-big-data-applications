Work is done in the movielens dataset found below:
https://grouplens.org/datasets/movielens/25m/

We're assuming a disk-based dataset, which means that the data is taken in chunks from the disk, and not loaded in the memory entirely, due
to its massive size.

K-means clustering is basic, as the dataset is only iterated through once and no convergence is achieved.
Collaborative filtering is done by calculating jaccard similarities between different users or items (movies). In the end, the user is recommended
20 movies based on his choice.

The user can choose how many clusters he wants to create, or what kind of collaborative filtering he wants to do.

**IMPORTANT:**
FileCreation.py must be run first, so some useful intermediate files can be created first.
