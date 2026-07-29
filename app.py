from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and encoders
model = joblib.load("models/career_model.pkl")
encoders = joblib.load("models/encoders.pkl")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    data = {}

    data["CGPA"] = float(request.form["cgpa"])

    data["Python"] = 1 if request.form["python"] == "Yes" else 0
    data["Java"] = 1 if request.form["java"] == "Yes" else 0
    data["SQL"] = 1 if request.form["sql"] == "Yes" else 0
    data["C++"] = 1 if request.form["cpp"] == "Yes" else 0
    data["HTML_CSS"] = 1 if request.form["html"] == "Yes" else 0

    data["Communication"] = request.form["communication"]
    data["Aptitude"] = request.form["aptitude"]

    data["Internship"] = 1 if request.form["internship"] == "Yes" else 0
    data["Certifications"] = 1 if request.form["certification"] == "Yes" else 0

    data["Interest"] = request.form["interest"]


    # Convert text values using encoders
    for col in ["Communication", "Aptitude", "Interest"]:
        data[col] = encoders[col].transform([data[col]])[0]


    input_data = pd.DataFrame([data])


    prediction = model.predict(input_data)[0]
    confidence = max(model.predict_proba(input_data)[0]) * 100
    confidence = round(confidence, 2)


    # Convert prediction back to career name
    career = encoders["Career"].inverse_transform([prediction])[0]


    career_details = {

    "AI Engineer": {
        "description": "AI Engineers design and develop intelligent systems using Artificial Intelligence and Machine Learning.",
        "skills": "Python, Machine Learning, Deep Learning, TensorFlow, Neural Networks",
        "courses": "Machine Learning, Deep Learning, Computer Vision, Natural Language Processing"
    },

    "Data Scientist": {
        "description": "Data Scientists analyze complex data and build predictive models to solve business problems.",
        "skills": "Python, Statistics, SQL, Data Visualization, Machine Learning",
        "courses": "Data Science, Statistics, Data Analytics, Machine Learning"
    },

    "Software Engineer": {
        "description": "Software Engineers design, develop and maintain software applications.",
        "skills": "Java, C++, Data Structures, Algorithms, Database",
        "courses": "DSA, System Design, Software Development"
    },

    "Web Developer": {
        "description": "Web Developers create and maintain websites and web applications.",
        "skills": "HTML, CSS, JavaScript, React, Backend Development",
        "courses": "Web Development, Full Stack Development"
    },

    "Cloud Engineer": {
        "description": "Cloud Engineers manage cloud infrastructure and deployment systems.",
        "skills": "Cloud Computing, Linux, Python, Networking",
        "courses": "AWS, Azure, Cloud Architecture"
    },

    "Cyber Security Analyst": {
        "description": "Cyber Security Analysts protect systems and networks from cyber threats.",
        "skills": "Networking, Linux, Ethical Hacking, Security Tools",
        "courses": "Cyber Security, Ethical Hacking, Network Security"
    },

    "Data Analyst": {
        "description": "Data Analysts convert raw data into meaningful insights.",
        "skills": "Python, SQL, Excel, Data Visualization",
        "courses": "Data Analytics, Power BI, SQL"
    },

    "Android Developer": {
        "description": "Android Developers create mobile applications for Android platforms.",
        "skills": "Java, Kotlin, Android Studio, APIs",
        "courses": "Android Development, Mobile App Development"
    }
}

    details = career_details.get(career)


    return render_template(
        "result.html",
        career=career,
        confidence=confidence,
        description=details["description"],
        skills=details["skills"],
        courses=details["courses"]
    )

if __name__ == "__main__":
    app.run(debug=True)