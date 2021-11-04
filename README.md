# Serving ML Models Using FastAPI  (Oauth2 and Logging Mechanism are Included)

This project includes serving a machine learning model using FastAPI with Outh2 security.



## Simplified Architecture

![Communication steps between client and FastAPI server](.gitbook/assets/image.png)

## &#x20;Step 1 - Creating Sqlite User Database in Python&#x20;

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



