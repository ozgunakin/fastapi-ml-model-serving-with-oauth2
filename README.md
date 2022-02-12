# Serving ML Models Using FastAPI  (Oauth2 and Logging Mechanism are Included)

This project includes serving a machine learning model using FastAPI with Outh2 security.



## Simplified Architecture

![Communication steps between client and FastAPI server](.gitbook/assets/image.png)

## Step 1 - Creating Sqlite User Database in Python&#x20;

We need to create user database to be able to authorize the user when the user request for access token. In this database we will store user-name and user-hashed-password (we will use Bcyrpt encrypted hash of the password)

* [x] Create username and password for the user

| username  | password     |
| --------- | ------------ |
| trialuser | trialpass123 |

* [x] Hash the password using online Bcrypt generator tool. ([https://bcrypt-generator.com](https://bcrypt-generator.com))

| password     | hashed-password                                              |
| ------------ | ------------------------------------------------------------ |
| trialpass123 | $2a$12$O/9CiF4Ul3WdEgDPCaYtt.r/QjA5kZORpZENxNkV4E8HuD/fZEnma |



* [x] Create a user sqlite database and insert username and hashed-password into the database using Python. (sqlitecreator.py)&#x20;

```
import sqlite3

conn = sqlite3.connect('usersdb') 
c = conn.cursor()

c.execute('''
          CREATE TABLE IF NOT EXISTS users
          ([username] TEXT, [full_name] TEXT,
          [email] TEXT, [hashed_password] TEXT, [disabled] BOOL)
          ''')
        
c.execute('''
          INSERT INTO users
                VALUES
                ('trialuser',"Trial User",
                "info@trialuser.com","$2a$12$O/9CiF4Ul3WdEgDPCaYtt.r/QjA5kZORpZENxNkV4E8HuD/fZEnma",
                "False")
          ''')

conn.commit()

print("finished")
```

## Step 2 - Developing Machine Learning Model

We will train a machine learning model using pima-indians-diabetes data (you can find the details here [https://www.kaggle.com/uciml/pima-indians-diabetes-database](https://www.kaggle.com/uciml/pima-indians-diabetes-database)). If you already have an ML model you can skip this step.

* [x] Open your jupyter notebook.

```
jupyter notebook
```

* [x] Create and dump dummy ML model ([https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/](https://machinelearningmastery.com/save-load-machine-learning-models-python-scikit-learn/))

```
# Save Model Using Pickle
import pandas
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
import pickle
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
names = ['preg', 'plas', 'pres', 'skin', 'test', 'mass', 'pedi', 'age', 'class']
dataframe = pandas.read_csv(url, names=names)
array = dataframe.values
X = array[:,0:8]
Y = array[:,8]
test_size = 0.33
seed = 7
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=test_size, random_state=seed)
# Fit the model on training set
model = LogisticRegression()
model.fit(X_train, Y_train)
# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
```

