import joblib
import pandas as pd

# Load các mô hình và vectorizer
vectorizer = joblib.load("models/vectorizer.pkl")
subtopic_model = joblib.load("models/subtopic_model.pkl")
topic_model = joblib.load("models/topic_model.pkl")

def predict_question(question):
    """
    Dự đoán subtopic và topic của câu hỏi.
    :param question: Câu hỏi cần dự đoán.
    :return: Subtopic và topic dự đoán.
    """
    # Chuyển câu hỏi thành vector
    question_vector = vectorizer.transform([question])
    
    # Dự đoán subtopic
    predicted_subtopic = subtopic_model.predict(question_vector)[0]
    
    # Dự đoán topic
    predicted_topic = topic_model.predict(question_vector)[0]
    
    return predicted_subtopic, predicted_topic

# Ví dụ sử dụng
if __name__ == "__main__":
    data_test = pd.read_csv("data/data_test.csv")
    is_true = 0
    for index, row in data_test.iterrows():
        question = row['question']
        topic = row["topic"]
        subtopic = row["subtopic"]
        print(f"question: {question}")
        print(f"topic: {topic}")
        print(f"subtopic: {subtopic}")
        # predict
        subtopic, topic = predict_question(question)
        if subtopic == subtopic and topic == topic:
            is_true += 1
        # print(f"Subtopic dự đoán: {subtopic}")
        # print(f"Topic dự đoán: {topic}")

    print(f"is_true: {is_true}")
    print(f"Accuracy: {is_true / len(data_test)}")
