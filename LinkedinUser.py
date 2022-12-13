import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import streamlit as st

s = pd.read_csv("social_media_usage.csv")


def clean_sm(x):
    binary_value = np.where(x.iloc[:,[1]]==1,1,0)
    x["Binary Value"] = binary_value
    print(x) 

df_dummy = pd.DataFrame({"Name":["John","Tom","Sam"], "Linkedin User?":[2,1,1]})
clean_sm(df_dummy)

ss = s.loc[:,["web1h","income","educ2","par","marital","gender","age"]]
ss.rename(columns={"web1h":"sm_li","educ2":"educ"},inplace=True)
ss["sm_li"] = np.where(ss["sm_li"]==1,1,0)
ss["educ"] = np.where(ss["educ"] <= 8,ss["educ"],np.nan)
ss["income"] = np.where(ss["income"] < 10,ss["income"],np.nan)
ss["age"] = np.where(ss["age"] < 99,ss["income"],np.nan)
ss["par"]= np.where(ss["par"] == 1,1,0)
ss["marital"] = np.where(ss["marital"] == 1,1,0)
ss["gender"] = np.where(ss["gender"] == 1,1,0)
ss = ss.dropna()
ss = ss.astype({"income":int,"educ":int,"age":int})


ss.corr()

y = ss["sm_li"]

X = ss[["income", "educ", "par","marital","gender","age"]]

# Split data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    stratify=y,       # same number of target in training & test set
                                                    test_size=0.2,    # hold out 20% of data for testing
                                                    random_state=500)


lr = LogisticRegression(class_weight="balanced")

lr.fit(X_train,y_train)

y_pred = lr.predict(X_test)

print(classification_report(y_test, y_pred))

print(confusion_matrix(y_test, y_pred))

pd.DataFrame(confusion_matrix(y_test, y_pred),
            columns=["Predicted negative", "Predicted positive"],
            index=["Actual negative","Actual positive"]).style.background_gradient(cmap="PiYG")

print(classification_report(y_test, y_pred))

newdata = pd.DataFrame({
    "income": [8, 8],
    "educ": [7,7],
    "par": [0, 0],
    "marital": [1,1],
    "gender": [0,0],
    "age": [42, 82],
    
})



newdata["sm_li"] = lr.predict(newdata)



# New data for features: income, educ, par, marital, gender,age
person = [8, 7, 0, 1, 0, 42]

# Predict class, given input features
predicted_class = lr.predict([person])

# Generate probability of positive class (=1)
probs = lr.predict_proba([person])

st.title("Using Machine Learning to Predict Linkedin Users")
st.subheader("By:Gaurav Patanker")

income_input = st.selectbox('Select Your Income Level', ['Less than $10,000', '10 to under $20,000','20 to under $30,000','30 to under $40,000', '40 to under $50,000',
'50 to under $75,00','75 to under $100,000','100 to under $150,000','$150,000 or more'])
st.markdown(f"You Selected: **{income_input}**")

education_input = st.selectbox('Select You highest level of education:',['Less than high school','High school incomplete','High school graduate',
'Some college, no degree','Two-year associate degree from a college or university','Four-year college or university degree/Bachelors degree',
'Some postgraduate or professional schooling, no postgraduate degree','Postgraduate or professional degree, including masters doctorate, medical or law degree, doctorate, medical or law degree'])
st.markdown(f"You Selected: **{education_input}**")

parent_input = st.selectbox('Are you a parent?',['True','False'])
st.markdown(f"You Selected: **{parent_input}**")

marital_input = st.selectbox('What is your Marital Status?',['Married','Not Married'])
st.markdown(f"You Selected: **{marital_input}**")

gender_input = st.selectbox('What is your gender',['Male','Female'])
st.markdown(f"You Selected: **{gender_input}**")

age_input = st.slider("What is your age?",0,98)
age_input2 = int(float(age_input))
st.markdown(f"You Selected: **{age_input2}**")

income_input2 = np.where(income_input == 'Less than $10,000',1,
                            (np.where(income_input == '10 to under $20,000',2,
                                (np.where(income_input == '20 to under $30,000',3,
                                    (np.where(income_input == '30 to under $40,000',4,
                                        (np.where(income_input == '40 to under $50,000',5,
                                            (np.where(income_input == '50 to under $75,00',6,
                                                (np.where(income_input == '75 to under $100,000',7,
                                                    (np.where(income_input == '100 to under $150,000',8,
                                                        (np.where(income_input == '$150,000 or more',9,0)))))))))))))))))

educ_input2 = np.where(education_input == 'Less than high school',1,
                            (np.where(education_input == 'High school incomplete',2,
                                (np.where(education_input == 'High school graduate',3,
                                    (np.where(education_input == 'Some college, no degree',4,
                                        (np.where(education_input == 'Two-year associate degree from a college or university',5,
                                            (np.where(education_input == 'Four-year college or university degree/Bachelors degree',6,
                                                (np.where(education_input == 'Some postgraduate or professional schooling, no postgraduate degree',7,
                                                    (np.where(education_input == 'Postgraduate or professional degree, including masters doctorate, medical or law degree, doctorate, medical or law degree',8,0)))))))))))))))

parent_input2 = np.where(parent_input == 'True',1,0)
marital_input2 = np.where(marital_input == 'Married',1,0)
gender_input2 = np.where(gender_input == 'Male',1,0)


input_df = [income_input2,educ_input2,parent_input2,marital_input2,gender_input2,age_input2]

newdata1 = pd.DataFrame({
    "income": [income_input2],
    "educ": [educ_input2],
    "par": [parent_input2],
    "marital": [marital_input2],
    "gender": [gender_input2],
    "age": [age_input2],
    
})
 

predicted_class2 = lr.predict(newdata1)
probs2 = lr.predict_proba(newdata1)
probs3 = np.round(probs2,4)*100
# Predict class, given input features
#predicted_class1 = (lr.predict([input_df]))

# Generate probability of positive class (=1)
#probs1 = lr.predict_proba([input_df])

#print(probs1)

st.markdown(f"Predicted class: {predicted_class2[0]}") # 0=not pro-environment, 1=pro-envronment
st.markdown(f"Probability that you are a Linkedin User: {probs3[0][1]}%")
#st.markdown(f"You Selected: **{educ_input2}**")

# import base64
# def add_bg_from_local(image_file):
#     with open(image_file, "rb") as image_file:
#         encoded_string = base64.b64encode(image_file.read())
#     st.markdown(
#     f"""
#     <style>
#     .stApp {{
#         background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
#         background-size: cover
#     }}
#     </style>
#     """,
#     unsafe_allow_html=True
#     )
# add_bg_from_local('test3.png')  
                                    