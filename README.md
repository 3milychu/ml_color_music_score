# ML Color Music Score

This repo provides instructions to take images of a music score and color code it by similarity using machine learning; the model fitted to the data uses kmeans.

Full guide: https://medium.com/nightingale/color-coding-music-scores-with-machine-learning-a2443ef27d96


## Raw data: /original-score

Directory containing the clean, unmarked electronic music score images for Fantasiestuckie Op. 73 which is used for the tutorial. Provided by Michael Schober (The New School for Social Research) and Neta Spiro (Royal College of Music).

## Image data: /images

Directory containing sliced images by beat from /original-score images

## Metadata: timestamps.csv

Metadata information of timing (measure, beat and timestamp in musical performance recording of the piece). Provided by Michael Schober (The New School for Social Research) and Neta Spiro (Royal College of Music)

## Model: k-means.ipynb

Python 3 notebook. Run through this notebook to output results of clustering image data and metadata using k-means. Will generate cluster_results.csv and cluster_results.json

Credits to Aaron Hill, Machine Learning Professor at Parsons School of Design for the code from which this notebook was built off of, as well as authoring my_measures.py: https://github.com/visualizedata/ml

## Data Visualization #1: results.html

html/css/js template to generate visualization of music score with color bars underneath

Example: https://3milychu.github.io/ml_color_music_score/results.html

## Data Visualization #2: cluster-analysis.html

Interactive visualization of beats grouped by named clusters

Example: https://3milychu.github.io/ml_color_music_score/cluster-analysis.html

