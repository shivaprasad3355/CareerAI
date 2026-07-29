import pandas as pd
import random

random.seed(42)

careers = [
    "AI Engineer",
    "Data Scientist",
    "Software Engineer",
    "Web Developer",
    "Cloud Engineer",
    "Cyber Security Analyst",
    "Data Analyst",
    "Android Developer"
]

communication = ["Poor", "Average", "Good", "Excellent"]
aptitude = ["Low", "Medium", "High"]

data = []

for i in range(500):
    career = random.choice(careers)

    row = {
        "CGPA": round(random.uniform(6.0, 9.8), 2),
        "Python": random.choice(["Yes", "No"]),
        "Java": random.choice(["Yes", "No"]),
        "SQL": random.choice(["Yes", "No"]),
        "C++": random.choice(["Yes", "No"]),
        "HTML_CSS": random.choice(["Yes", "No"]),
        "Communication": random.choice(communication),
        "Aptitude": random.choice(aptitude),
        "Internship": random.choice(["Yes", "No"]),
        "Certifications": random.choice(["Yes", "No"]),
        "Interest": career,
        "Career": career
    }

    data.append(row)

df = pd.DataFrame(data)

df.to_csv("career_guidance_dataset_500.csv", index=False)

print("Dataset created successfully!")