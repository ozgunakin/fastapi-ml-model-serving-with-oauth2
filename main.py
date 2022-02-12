#Libraries for FastAPI
from typing import Optional,Set,List
from fastapi import FastAPI,Request
from transformers import AutoTokenizer, AutoModel,BertTokenizer,BertForSequenceClassification,pipeline
import numpy as np
import json
from typing import Optional
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel
import pickle


#Libraries for userdb connection
import sqlite3
import pandas as pd
import json

#Libraries for rate limit
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

#Libraries for logging
from datetime import datetime


#PARAMETERS and DESCRIPTIONS
# to get a string like this run:
# openssl rand -hex 32
SECRET_KEY = "43d3ac5e3b6bdf95225ef0abfb3a271e3784055a31f6d2bc1d0efdfd5e29fdf7"
ALGORITHM = "HS256"
#Oauth2 
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

'''
#Ml-model
model_name = "logistic"
model = BertForSequenceClassification.from_pretrained("iyi bert/outputs/checkpoint-4732-epoch-2")
tokenizer = AutoTokenizer.from_pretrained("iyi bert/outputs/checkpoint-4732-epoch-2")
classifier = pipeline("text-classification",model,tokenizer=tokenizer)
'''

#Apı limitter description
limiter = Limiter(key_func=get_remote_address)

#CLASS DEFINITIONS
class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: Optional[str] = None

class User(BaseModel):
    username: str
    email: Optional[str] = None
    full_name: Optional[str] = None
    disabled: Optional[bool] = None

class UserInDB(User):
    hashed_password: str


#USER DEFINED FUNCTIONS
def import_users_db():
    conn = sqlite3.connect('usersdbtrial') 
    c = conn.cursor()
    c.execute('''
          SELECT *
          FROM users
          ''')

    df = pd.DataFrame(c.fetchall(),columns=["username","full_name",
          "email","hashed_password",
          "disabled"])
    print (df)
    df.set_index(df["username"],inplace=True)
    result = df.to_json(orient="index")
    users_db = json.loads(result)

    return users_db

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return UserInDB(**user_dict)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
        return False
    return user

def create_access_token(data: dict):
    to_encode = data.copy()
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception
    user = get_user(users_db, username=token_data.username)
    if user is None:
        raise credentials_exception
    return user

async def get_current_active_user(current_user: User = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

def log_api_calls(request_data,response_data,request_header):
    now = datetime.now()
    log_data={"date":str(now),"request":str(request_data),"response":str(response_data),"request_header":str(request_header)},
    df_log=pd.DataFrame(log_data)
    df_log.to_csv("log.csv", mode="a", header=False, index=False)

    message="logging succesfull"
    print(message)

#APP STARTPOINT
#app = FastAPI(docs_url=None, redoc_url=None)
app = FastAPI()
#Add api limitter to app
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

#Userdb connection
users_db = import_users_db() 

@app.post("/token", response_model=Token, tags=["create-token"])
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(users_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = create_access_token(
        data={"sub": user.username}
    )
    return {"access_token": access_token, "token_type": "bearer"}


@app.get("/v1/test",tags=["connection-test"])
async def test_api(current_user: User = Depends(get_current_active_user)):
    return {"Hello": "World"}

'''
@app.post("/v1/sentiment_v1",tags=["ml-models"])
#@limiter.limit("2/minute")
async def predict(request: Request,current_user: User = Depends(get_current_active_user),status_code=status.HTTP_200_OK):

    request_data = await request.json()
    response_data = request_data.copy()
    customer=[d['Text'] for d in request_data['Channels'][0]['SpeechRegions']]
    agent=[d['Text'] for d in request_data['Channels'][1]['SpeechRegions']]
    response_data['Channels'][0]['Sentiments']=classifier(customer)
    response_data['Channels'][1]['Sentiments']=classifier(agent)

    #logging
    request_header=request.headers
    log_api_calls(request_data,response_data,request_header)

    return response_data
'''

@app.post("/regression")
def postanitem(inp: int):
    inp = np.array(inp).reshape(-1,1)
    filename = 'finalized_model.sav'
    loaded_model = pickle.load(open(filename, 'rb'))
    out = loaded_model.predict(inp)
    print(float(out))
    o = {'Output':float(out)}
    return o