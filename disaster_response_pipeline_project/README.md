# Disaster Response Pipeline Project
This project is a machine learning pipeline to categorize messages sent during disaster events. It includes a web app where an emergency worker can input a new message and get classification results in several categories. The web app also displays visualizations of the training data.


### Instructions:
1. [Download the training datasets](https://drive.google.com/open?id=1UF2YPf4qin0pMM_HYnMLeCq7tK8jyYA9) and place them in data folder

2. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

4. Go to http://0.0.0.0:3001/


### Screenshots:
Once a user message is submitted, its labels are predicted by a pre-trained
Random Forest Classifier:
![classification Task](https://raw.githubusercontent.com/claass/udacity_datascientist_projects/master/disaster_response_pipeline_project/screenshots/classification_task.png)
