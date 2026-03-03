# Computational generics
This repository contains information for the project computational generics. It builds off of [a previous version of this project](https://github.com/markkho/generics-learning/).

If you have any questions, please contact me (Marianna Zhang) at <marianna.zhang@nyu.edu>.

# Studies

* **Study 6**: How do generic versus specific statements affect a category's inductive potential?
* **Study 8**: Does inductive potential vary based on the proportion of generic versus specific statements heard? 
* **Study 9**: Does the generalization of features across a category based on the similarity between the features and known category features? 

# Repo structure

## Writeups
Posters, conference proceedings (e.g., [CogSci paper](https://escholarship.org/content/qt2rs3j5vq/qt2rs3j5vq.pdf)), slides, etc. 

## Model
Model code written in Python (.py) and Jupyter Notebooks (.ipynb) to fit model parameters, simulate responses, and visualize model simulated responses. 

### App
Contains a streamlit app to play with different model parameters and generate predictions. To run the app:
1. Fork this repo.
2. Install [streamlit](https://docs.streamlit.io/get-started/installation) and required packages, such as [gorgo](https://github.com/markkho/gorgo).
3. Run the app locally:
```
streamlit run models/app/app.py
```

## Materials
Qualtrics survey files (.qsf) for each study. 

## Data
Raw data (.csv) exported from Qualtrics for each study. 

## Analyses
RMarkdown notebooks (.Rmd) and their knitted versions (.html) for visualizing and analyzing behavioral responses, and comparing with model simulated responses, for each study. 