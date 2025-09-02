import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

# Download required NLTK resources (run once)
nltk.download('punkt')
nltk.download('stopwords')

ps = PorterStemmer()

# Preprocessing function
def transform_text(text):
    text = text.lower()                          # lowercase
    text = nltk.word_tokenize(text)              # tokenize
    
    y = []
    for i in text:
        if i.isalnum():                          # keep only alphanumeric
            if i not in stopwords.words('english') and i not in string.punctuation:
                y.append(ps.stem(i))             # remove stopwords + punctuation + stem
    
    return " ".join(y)



# Load trained vectorizer & model
try:
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
except FileNotFoundError:
    st.error(" Model files not found! Please run the training script first.")
    st.stop()
except Exception as e:
    st.error(f" Error loading model files: {e}")
    st.stop()

# Streamlit UI
st.title('📧 Email / SMS Spam Classifier')

input_sms = st.text_area("✍️ Enter the message here:")

if st.button("Predict"):
    if input_sms.strip() == "":
        st.warning(" Please enter a message to classify.")
    else:
        # 1. Preprocess
        transformed_sms = transform_text(input_sms)

        # 2. Vectorize
        vector_input = tfidf.transform([transformed_sms])

        # 3. Predict
        result = model.predict(vector_input)[0]

        # 4. Display
        if result == 1:
            st.error(" Spam Message Detected!")
        else:
            st.success(" This is Not Spam")
