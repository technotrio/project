# project
MLOPs Final Project
Created a data generating script.
Generated a .csv file.
Initialised DVC.
Utilizing DVC to track the csv file.


# how to clone the changes of i200588 

Clone repository:
 - git clone https://github.com/technotrio/project.git

Navigate to git branch:
 - git checkout i200828

Make your own virtual environment and activate it:
 - python -m venv venv 
 - venv\Scripts\activate 

Install Dependencies: 
 - pip install -r requirements.txt

Initialize DVC and Fetch Data:
 - dvc pull

(Select the google drive account that has access to the remote files, and allow the permissions once redirected to web page,
 it can be done through terminal too )

Stash Local Changes before switching Branch
 - git stash
 - git checkout <branch-name>
 - git stash apply



