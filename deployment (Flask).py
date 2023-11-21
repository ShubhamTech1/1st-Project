from flask import Flask, render_template, request
import re
import pandas as pd
import numpy as np
import copy
import joblib, pickle
from sqlalchemy import create_engine
  
# creating Engine which connect to mysql
user = 'user1' # user name
pw = 'user1' # password
db = 'salary_db' # database
  
engine = create_engine(f"mysql+pymysql://{user}:{pw}@localhost/{db}")

db = create_engine(engine)
conn = db.connect()

# Load the saved model
model = joblib.load('processed1')

clean = joblib.load('clean_NB')
mnb = pickle.load(open('multinomialNB.pkl', 'rb'))

# Define flask
app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        test = pd.read_csv(r"D:\DATA SCIENTIST\DATA SCIENCE\ASSIGNMENTS\SUPERVISED LEARNING\CLASSIFICATION\naive bayes\SalaryData_Test.csv")    
 
        test_clean = pd.DataFrame(clean.transform(test), columns=clean.get_feature_names_out())
        y_test_pred = mnb.predict(test_clean)

        final = pd.concat([test, y_test_pred], axis = 1)

        final.to_sql('salary_predictions', con = conn, if_exists = 'replace', index= False)
        conn.autocommit = True
               
        return render_template("new.html", Y = final.to_html(justify = 'center'))

if __name__=='__main__':
    app.run(debug = True)
