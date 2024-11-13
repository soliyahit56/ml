import streamlit as st
import pickle

model = pickle.load(open('spam.pkl','rb'))
cv = pickle.load(open('vectorize.pkl','rb'))

st.title("Email Spam Classification Application")
st.write("this is a machine learning applicatiom to classification email as spam and ham")
user_input= st.text_area("enter an email to classifiction",height=150)
if st.button("classify"):
    if user_input:
        data=[user_input]
        vect=cv.transform(data).toarray()
        pred=model.predict(vect)
        if pred[0]==0:
            st.success("this email is not spem ")
        else:
            st.error("this email is spem")

# streamlit run app.py
#pip install -r requirements.txt
#pythom -m venv myenv 
#.\myenv\Scripts\activate 
