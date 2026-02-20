# Practical 3 – Spam Email Detector

import streamlit as st

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# ---------------------------------
# Title
# ---------------------------------
st.title("Spam Email Detector")


# ---------------------------------
# Sidebar settings
# ---------------------------------
st.sidebar.header("Settings")

test_size = st.sidebar.slider(
    "Test Size",
    min_value=0.10,
    max_value=0.50,
    value=0.25,
    step=0.05
)

c_value = st.sidebar.slider(
    "Regularization (C)",
    min_value=0.1,
    max_value=5.0,
    value=1.0,
    step=0.1
)


# ---------------------------------
# Sample dataset
# ---------------------------------
emails = [
    "Win a free iPhone now",
    "Meeting at 11 am tomorrow",
    "Congratulations you won lottery",
    "Project discussion with team",
    "Claim your prize immediately",
    "Please find the attached report",
    "Limited offer buy now",
    "Urgent offer expires today",
    "Schedule the meeting for Monday",
    "You have won a cash prize",
    "Monthly performance report attached",
    "Exclusive deal just for you"
]

# 1 = spam, 0 = not spam
labels = [1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1]


# ---------------------------------
# Vectorizer
# ---------------------------------
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    min_df=1
)

X = vectorizer.fit_transform(emails)


# ---------------------------------
# Train test split
# ---------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    labels,
    test_size=test_size,
    random_state=42,
    stratify=labels
)


# ---------------------------------
# Model
# ---------------------------------
model = LinearSVC(C=c_value, random_state=42)
model.fit(X_train, y_train)


# ---------------------------------
# Accuracy
# ---------------------------------
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

st.write("Model Accuracy :", round(acc, 3))


# ---------------------------------
# User input
# ---------------------------------
st.subheader("Check your Email")

user_msg = st.text_area("Enter Email Message")

if st.button("Check"):

    if user_msg.strip() == "":
        st.warning("Please enter a message.")
    else:
        msg_vec = vectorizer.transform([user_msg])
        pred = model.predict(msg_vec)[0]

        if pred == 1:
            st.error("Result : Spam Email")
        else:
            st.success("Result : Not a Spam Email")
