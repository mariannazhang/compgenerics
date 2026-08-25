# Learning about the inductive potential of categories from language
This repository contains experiment materials, data, analyses, and figures for "Learning about the inductive potential of categories from language". It builds off of [a previous version of this project](https://github.com/markkho/generics-learning/).

If you have any questions, please contact me (Marianna Zhang) at <marianna.zhang@nyu.edu>.

# Studies & preregistrations

* **Study 1** (labeled study 6 internally): How do generic versus specific statements affect the category's inferred inductive potential? ([preregistration](https://osf.io/htzra/overview))

* **Study 2** (labeled study 8 internally): Does inductive potential vary based on the proportion of generic versus specific statements heard? ([preregistration](https://osf.io/vhn4j/overview))

* **Study 3** (labeled study 9 internally): Does the generalization of features across a category based on the similarity between the features and known category features? ([preregistration](https://osf.io/d9mwv/overview))


# Repository structure

```
├── model
│   ├── model-gp
│   └── ...
├── materials
│   └── study6_prereg_survey.qsf
├── data
│   ├── study6_prereg.csv
│   └── ...
├── analyses
│   └── study 6 prereg
│       └── figs
│   └── ...
├── writeups
```

## Model
Model code written in Python (.py) and Jupyter Notebooks (.ipynb) to fit model parameters, simulate responses, and visualize model simulated responses. 

## Materials
Qualtrics survey files (.qsf) for each study. 

## Data
Data (.csv) exported from Qualtrics with Prolific IDs hashed for each study. 

## Analyses
RMarkdown notebooks (.Rmd) and their knitted versions (.html) for visualizing and analyzing behavioral responses, and comparing with model simulated responses, for each study. 

## Writeups
Posters, conference proceedings (e.g., [CogSci paper](https://escholarship.org/content/qt2rs3j5vq/qt2rs3j5vq.pdf)), slides, etc. 
