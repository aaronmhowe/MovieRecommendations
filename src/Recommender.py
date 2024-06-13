# import numpy as np
# from numpy import linalg
# from itertools import combinations
import time
import os
import csv
import math

############################################################################
## This program implements the item-item collaborative filtering          ##
## algorithm for recommender systems, such as those used to recommend     ##
## items to users of applications developed by Netflix, Amazon, etc.      ##
## @author Aaron Howe                                                     ##
## @version Python 3.10.12                                                ##
############################################################################

# path references to each .csv file in 'movie-lens-data'
in_movies = os.path.join(os.path.dirname(__file__), 'movie-lens-data/movies.csv')
in_ratings = os.path.join(os.path.dirname(__file__), 'movie-lens-data/ratings.csv')
in_links = os.path.join(os.path.dirname(__file__), 'movie-lens-data/links.csv')
in_tags = os.path.join(os.path.dirname(__file__), 'movie-lens-data/tags.csv')

# variable used to dump application output into a .txt file
out = "./output.txt"

neighborhood_size = 5
# variable that defines the max number of recommendations per user
rec_count = 5


# function that reads and returns the data contained within each .csv file in 'movie-lens-data'
def read_files(movies, ratings, links, tags):

    movies_csv = {}
    ratings_csv = {}
    links_csv = {}
    tags_csv = {}

    # reads the data in movies.csv
    with open(movies, 'r', encoding='utf-8') as file:
        file_reader = csv.DictReader(file)
        for data in file_reader:
            movie_id = int(data['movieId'])
            movies_csv[movie_id] = {'title': data['title'], 'genres': data['genres'].split('|')}

    # reads the data in ratings.csv
    with open(ratings, 'r', encoding='utf-8') as file:
        file_reader = csv.DictReader(file)
        for data in file_reader:
            user_id = int(data['userId'])
            movie_id = int(data['movieId'])
            rating = float(data['rating'])
            timestamp = int(data['timestamp'])
            # zero case
            if user_id not in ratings_csv:
                ratings_csv[user_id] = {}
            ratings_csv[user_id][movie_id] = {'rating': rating, 'timestamp': timestamp}

    # reads the data in links.csv
    with open(links, 'r', encoding='utf-8') as file:
        file_reader = csv.DictReader(file)
        for data in file_reader:
            movie_id = int(data['movieId'])
            imdB_id = data['imdbId']
            tmdB_id = data['tmdbId']
            links_csv[movie_id] = {'imdbId': imdB_id, 'tmdbId': tmdB_id}

    # reads the data in tags.csv
    with open(tags, 'r', encoding='utf-8') as file:
        file_reader = csv.DictReader(file)
        for data in file_reader:
            user_id = int(data['userId'])
            movie_id = int(data['movieId'])
            tag = data['tag']
            timestamp = int(data['timestamp'])
            if movie_id not in tags_csv:
                tags_csv[movie_id] = {}
            if user_id not in tags_csv[movie_id]:
                tags_csv[movie_id][user_id] = []
            tags_csv[movie_id][user_id].append({'tag': tag, 'timestamp': timestamp})

    return movies_csv, ratings_csv, links_csv, tags_csv


# constructing item profiles that define their characteristics based on user
# interaction with them. These item profiles will be used to compute similarities
def profiles(movies, ratings, tags):

    # storing item profiles as a list
    item_profiles = {}

    # movie data in item profiles (title, genre, etc.)
    for movie_id, movie_data in movies.items():
        item_profiles[movie_id] = {'title': movie_data['title'], 'genres': movie_data['genres'], 'avg_rating': 0.0, 'ratings': {}, 'tags': []}
    
    # data for movie ratings in item profiles
    for user_id, user_ratings in ratings.items(): # for ratings from users
        for movie_id, movie_ratings in user_ratings.items(): # for ratings of movies
            rating = movie_ratings['rating'] # initialize rating
            item_profiles[movie_id]['ratings'][user_id] = rating  # add rating to item profile

    # data for rating averages
    for movie_id in item_profiles:
        ratings = item_profiles[movie_id]['ratings']
        if ratings:
            avg = sum(ratings.values()) / len(ratings)
            item_profiles[movie_id]['avg_rating'] = avg

    # data for tags applied to movies in item profiles
    for movie_id, tag_data in tags.items(): # for tags of movies
        for user_id, tags in tag_data.items(): # for tags from users
            for data in tags: 
                tag = data['tag'] # initialize tag
                item_profiles[movie_id]['tags'].append(tag) # add tag to the item profile

    return item_profiles 



def compute_sim_score(item_profiles):

    # list to store computed similarity scores
    similarity_scores = {}
    # list to store rating averages
    averages = {}
    
    # computing rating averages per item
    for item_id, item_profile in item_profiles.items():
        ratings = item_profile['ratings']
        if ratings:
            averages[item_id] = sum(ratings.values()) / len(ratings)
        else:
            averages[item_id] = 0.0

    # computing similarity scores between rated items using their centered cosine similarity
    for item_1 in item_profiles:
        similarity_scores[item_1] = {}
        for item_2 in item_profiles:
            if item_1 != item_2:
                # compute similarity score and store it in the list
                similarity_score = cosine_similarity(item_profiles[item_1], item_profiles[item_2], averages[item_1], averages[item_2])
                similarity_scores[item_1][item_2] = similarity_score

    # return the list of similarity scores
    return similarity_scores


# function to compute the centered cosine similarity between two items
def cosine_similarity(item_1, item_2, mean_1, mean_2):

    # set of users and their ratings
    users = set(item_1['ratings'].keys()) & set(item_2['ratings'].keys())
    if not users:
        return 0.0
    
    # variables to be computed in the cosine-similarity formula
    # dot_product
    # euclidean_length_item_1
    # euclidean_length_item_2
    
    # computing the dot product of item 1 and item 2
    # for user_id in users:
    dot_product = sum((item_1['ratings'][user_id] - mean_1) *
                      (item_2['ratings'][user_id] - mean_2) for user_id in users)
    
    # computing item 1's euclidean length
    # for user_id in item_1['ratings']:
    euclidean_length_item_1 = math.sqrt(sum((item_1['ratings'][user_id] - mean_1) ** 2
                                        for user_id in item_1['ratings']))

    # computing item 2's euclidean length
    # for user_id in item_2['ratings']:
    euclidean_length_item_2 = math.sqrt(sum((item_2['ratings'][user_id] - mean_2) ** 2
                                        for user_id in item_2['ratings']))
    
    # computing the dot product of the euclidean lengths of item 1 and item 2
    euclidean_dot_product = euclidean_length_item_1 * euclidean_length_item_2
    if euclidean_dot_product == 0.0:
        return 0.0

    # computing and returning the cosine similarity
    return dot_product / euclidean_dot_product


# function to construct the subset of items that are considered the most similar to another item
def get_neighborhood_set(sim_score, size):
    
    neighborhood_set = {}

    for item_id in sim_score:
        # sorting the items in descending order from most similar to least similar
        sorted_items = sorted(sim_score[item_id].items(), key=lambda x: x[1], reverse=True)
        # n = the top-N similar items in the subset
        n = [item for item, _ in sorted_items[:size]]
        # appending ties
        if len(sorted_items) > size:
            i = sorted_items[size - 1][1]
            for item, item_score in sorted_items[size:]:
                if item_score == i:
                    n.append(item)
                else:
                    break

        # ensuring lexicographic sorting 
        n = sorted(n)
        neighborhood_set[item_id] = n
    
    return neighborhood_set


# function to predict the ratings for each user-item pair
def estimate_ratings(item_profiles, neighborhood):

    estimated_ratings = {}

    for item_id, item_profile in item_profiles.items():
        # top-n similar items
        n = neighborhood[item_id]
        for user_id in item_profile['ratings']:
            if user_id not in estimated_ratings:
                estimated_ratings[user_id] = {}
            if item_id not in estimated_ratings[user_id]:
                rated_items = [item for item in neighborhood if user_id in item_profiles[item]['ratings']]
                if not rated_items:
                    return None
                # rating scores summed
                computed_sum = sum(item_profiles[item]['ratings'][user_id] for item in rated_items)
                # summing the total number of ratings given, and summing them using the max-value
                size = len(rated_items * neighborhood_size)
                estimate = computed_sum / size
                estimated_ratings[user_id][item_id] = estimate

    return estimated_ratings


# creating recommendations for users based on computed similarities and estimated ratings
def recommend_items(estimated_ratings, count):

    recommendations = {}

    for user_id in estimated_ratings:
        ratings = sorted(estimated_ratings[user_id].items(), key=lambda x: x[1], reverse=True)
        # top-N items to recommend to users
        n = [item_id for item_id, _ in ratings[:count]]

        ## This block of code appears to be causing the function to ignore the count = 5.
        ## Likely due to appending a matching item_id
        ## the output looks fine without it, though.
        # # appending ties
        # if len(ratings) > count:
        #     # checking if the next item over is a tie, and appending if so
        #     i = ratings[count - 1][1]
        #     for item_id, rating in ratings[count:]:
        #         if rating == i:
        #             n.append(item_id)
        #         else:
        #             break

        # sorting the items lexicographically
        n = sorted(n)
        # add the recommended items to the list
        recommendations[user_id] = n 
    
    return recommendations


# function to dump the output into output.txt
def dump_output(items, output):

    with open(output, 'w') as file:
        for user_id in sorted(items.keys()):
            recommended_items = items[user_id]
            write_out = f"{user_id}"
            for item_id in recommended_items:
                write_out += f" {item_id}"
            file.write(write_out + "\n")


# function that measures and prints the run-time of the program execution
# this function is extra-curricular and is not needed to successfully execute this program
def run_time(timer):

    def wrapper(*args, **kwargs):

        start = time.time()
        result = timer(*args, **kwargs)
        end = time.time()
        total = end - start
        print(f"Program Run-Time: {total:.2f} seconds")
        return result

    return wrapper


# main application function
@run_time
def main():

    # loading csv data into memory
    movies, ratings, links, tags = read_files(in_movies, in_ratings, in_links, in_tags)

    # constructing the item profiles
    item_profiles = profiles(movies, ratings, tags)
    # computing the similarity scores
    similarity_scores = compute_sim_score(item_profiles)
    # constructing neighborhood sets
    neighborhood_set = get_neighborhood_set(similarity_scores, neighborhood_size)
    # predicting ratings that users might give
    estimated_ratings = estimate_ratings(item_profiles, neighborhood_set)
    # creating recommendations
    item_recommendations = recommend_items(estimated_ratings, rec_count)
    # dumping the output into output.txt
    dump_output(item_recommendations, out)


    # quick print that serves to alert that the program executed successfully
    print(f"Houston, output has landed in {out}")


if __name__ == "__main__":
    main()
