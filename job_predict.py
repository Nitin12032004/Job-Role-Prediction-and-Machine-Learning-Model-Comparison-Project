import streamlit as st
import pandas as pd
import itertools
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

st.title("Job Role Prediction")

@st.cache_resource
def load_and_train():
    data = pd.read_csv("Merged_data2.csv")
    df = pd.DataFrame(data)
    df['skill_count'] = df['combined_skills'].apply(len)
    scaler = StandardScaler()
    df['salary_normalized'] = scaler.fit_transform(df[['average_salary_value']])
    X = df[['combined_skills', 'salary_normalized', 'average_experience', 'skill_count']]
    y = df['job_title']
    X['combined_skills'] = X['combined_skills'].apply(lambda x: ' '.join(x))
    X = pd.concat([X.drop('combined_skills', axis=1),
                   X['combined_skills'].str.get_dummies(sep=' ')],
                  axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return scaler, X, model

scaler, X_all, model = load_and_train()

st.write("Enter your details below:")

skills_input = st.text_input("Enter your skills (separated by commas)")
experience_input = st.number_input("Enter your years of experience", min_value=0.0, value=0.0)
salary_input = st.number_input("Enter your expected annual salary", min_value=200000.0, step=1000.0, value=200000.0)

if st.button("Predict Job Roles"):
    user_skills = [skill.strip() for skill in skills_input.split(",") if skill.strip() != ""]
    if len(user_skills) == 0:
         st.error("Please enter at least one skill.")
    else:
         # Generate all non-empty subsets of the entered skills.
         subsets = []
         for r in range(1, len(user_skills) + 1):
             subsets.extend(itertools.combinations(user_skills, r))
         
         predicted_jobs = set()
         
         # Generate salary values from the entered salary down to 200,000 (decrement 100,000)
         salary_values = []
         current_salary = salary_input
         while current_salary >= 200000:
             salary_values.append(current_salary)
             current_salary -= 100000
         
         # Generate experience values from the entered experience (integer) down to 0
         exp_values = list(range(int(experience_input), -1, -1))
         
         # Loop over every combination of skill subset, salary, and experience
         for subset in subsets:
             skill_string = " ".join(subset)
             for salary_val in salary_values:
                 norm_salary = scaler.transform([[salary_val]])[0][0]
                 for exp_val in exp_values:
                     skill_count = len(subset)
                     row = pd.DataFrame([[skill_string, norm_salary, exp_val, skill_count]],
                                        columns=['skills', 'salary_normalized', 'average_experience', 'skill_count'])
                     row = pd.concat([row.drop('skills', axis=1),
                                      row['skills'].str.get_dummies(sep=' ')],
                                     axis=1)
                     row = row.reindex(columns=X_all.columns, fill_value=0)
                     pred = model.predict(row)[0]
                     predicted_jobs.add(pred)
         
         if predicted_jobs:
             st.success("Predicted Job Roles: " + ", ".join(predicted_jobs))
         else:
             st.write("No job roles predicted.")
