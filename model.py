import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pickle
import os
import kagglehub

class InternPerformancePredictor:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.important_features = [
            "Internships_Completed", "Projects_Completed", "Soft_Skills_Score",
            "Networking_Score", "Starting_Salary", "Career_Satisfaction",
            "Work_Life_Balance", "GPA_Level", "Uni_GPA_Level", "Experience_Score"
        ]
        self.label_map = {0: "Low Performer", 1: "Moderate Performer", 2: "High Performer"}

    def load_dataset_from_kaggle(self):
        path = kagglehub.dataset_download("adilshamim8/education-and-career-success")
        for filename in os.listdir(path):
            if filename.endswith(".csv"):
                csv_file_path = os.path.join(path, filename)
                break
        else:
            raise FileNotFoundError("No CSV file found.")
        df = pd.read_csv(csv_file_path)

        df.rename(columns={
            "Internships": "Internships_Completed",
            "Projects": "Projects_Completed",
            "SoftSkills": "Soft_Skills_Score",
            "Networking": "Networking_Score",
            "StartingSalary": "Starting_Salary",
            "CareerSatisfaction": "Career_Satisfaction",
            "WorkLifeBalance": "Work_Life_Balance",
            "HighSchoolGPA": "High_School_GPA",
            "UniversityGPA": "University_GPA"
        }, inplace=True)

        df["GPA_Level"] = pd.cut(df["High_School_GPA"], bins=[0, 4, 7, 10], labels=[0, 1, 2])
        df["Uni_GPA_Level"] = pd.cut(df["University_GPA"], bins=[0, 4, 7, 10], labels=[0, 1, 2])
        df["Experience_Score"] = df["Internships_Completed"] + df["Projects_Completed"]
        df["Intern_Performance_Score"] = (
            df["Internships_Completed"] + df["Projects_Completed"] +
            df["Certifications"] + df["Soft_Skills_Score"] + df["University_GPA"]
        )
        df["Performance_Class"] = pd.cut(df["Intern_Performance_Score"], bins=[0, 8, 15, 25], labels=[0, 1, 2])
        return df

    def preprocess_data(self, df):
        df.dropna(inplace=True)
        df = df[self.important_features + ["Performance_Class"]]
        df.dropna(subset=["Performance_Class"], inplace=True)
        return df

    def train_model(self):
        df = self.load_dataset_from_kaggle()
        df = self.preprocess_data(df)

        X = df[self.important_features]
        y = df["Performance_Class"]

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        models = {
            "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42),
            "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
            "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
            "Support Vector Machine": SVC(probability=True, kernel='rbf', C=1.0, random_state=42),
            "Naive Bayes": GaussianNB()
        }

        best_accuracy = 0
        best_model = None
        best_model_name = ""

        print("Model Comparison:\n")
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            print(f"{name}:")
            print(f"  Accuracy : {acc:.4f}")
            print(f"  Precision: {prec:.4f}")
            print(f"  Recall   : {rec:.4f}")
            print(f"  F1-Score : {f1:.4f}\n")

            if acc > best_accuracy:
                best_accuracy = acc
                best_model = model
                best_model_name = name

        print(f"\nBest Model: {best_model_name} (Accuracy: {best_accuracy:.4f})")
        self.model = best_model
        return best_accuracy

    def save_model(self):
        if self.model is None or self.scaler is None:
            raise ValueError("Model must be trained before saving")
        pickle.dump(self.model, open("model.pkl", "wb"))
        pickle.dump(self.scaler, open("scaler.pkl", "wb"))
        pickle.dump(self.important_features, open("features.pkl", "wb"))
        print("Model saved successfully!")

    def load_model(self):
        try:
            self.model = pickle.load(open("model.pkl", "rb"))
            self.scaler = pickle.load(open("scaler.pkl", "rb"))
            self.important_features = pickle.load(open("features.pkl", "rb"))
            print("Model loaded successfully!")
            return True
        except FileNotFoundError:
            print("Model files not found.")
            return False

    def predict(self, input_data):
        if self.model is None or self.scaler is None:
            if not self.load_model():
                self.train_model()
                self.save_model()

        gpa_level = self._categorize_gpa(input_data['high_school_gpa'])
        uni_gpa_level = self._categorize_gpa(input_data['university_gpa'])
        experience_score = input_data['internships'] + input_data['projects']

        features = {
            "Internships_Completed": input_data['internships'],
            "Projects_Completed": input_data['projects'],
            "Soft_Skills_Score": input_data['soft_skills'],
            "Networking_Score": input_data['networking'],
            "Starting_Salary": input_data['salary'],
            "Career_Satisfaction": input_data['satisfaction'],
            "Work_Life_Balance": input_data['work_life_balance'],
            "GPA_Level": gpa_level,
            "Uni_GPA_Level": uni_gpa_level,
            "Experience_Score": experience_score
        }

        user_df = pd.DataFrame([features])[self.important_features]
        user_scaled = self.scaler.transform(user_df)

        pred_proba = self.model.predict_proba(user_scaled)[0]
        pred_label = pred_proba.argmax()

        result = {
            'prediction': self.label_map[pred_label],
            'confidence': float(pred_proba[pred_label] * 100),
            'probabilities': {
                self.label_map[i]: float(pred_proba[i] * 100) for i in range(len(pred_proba))
            }
        }
        return result

    def _categorize_gpa(self, gpa):
        if gpa <= 4:
            return 0
        elif gpa <= 7:
            return 1
        else:
            return 2

if __name__ == "__main__":
    predictor = InternPerformancePredictor()
    if not predictor.load_model():
        predictor.train_model()
        predictor.save_model()