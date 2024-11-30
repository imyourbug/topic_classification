import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. Đọc dữ liệu
data_path = "data/questions_topics_subtopic.csv"  # File dataset
data = pd.read_csv(data_path)

# 2. Tiền xử lý
X = data['question']  # Câu hỏi
y_subtopic = data['subtopic']  # Subtopic
y_topic = data['topic']  # Topic

# Chuyển đổi câu hỏi thành vector
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# 3. Chia dữ liệu cho subtopic
X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split(X_tfidf, y_subtopic, test_size=0.2, random_state=42)

# Huấn luyện mô hình subtopic
subtopic_model = MultinomialNB()
subtopic_model.fit(X_train_sub, y_train_sub)

# Đánh giá mô hình subtopic
print("Subtopic Model Accuracy:", accuracy_score(y_test_sub, subtopic_model.predict(X_test_sub)))
print("Subtopic Classification Report:")
print(classification_report(y_test_sub, subtopic_model.predict(X_test_sub)))

# 4. Chia dữ liệu cho topic
X_train_topic, X_test_topic, y_train_topic, y_test_topic = train_test_split(X_tfidf, y_topic, test_size=0.2, random_state=42)

# Huấn luyện mô hình topic
topic_model = MultinomialNB()
topic_model.fit(X_train_topic, y_train_topic)

# Đánh giá mô hình topic
print("Topic Model Accuracy:", accuracy_score(y_test_topic, topic_model.predict(X_test_topic)))
print("Topic Classification Report:")
print(classification_report(y_test_topic, topic_model.predict(X_test_topic)))

# 5. Lưu mô hình và vectorizer
joblib.dump(subtopic_model, "models/subtopic_model.pkl")
joblib.dump(topic_model, "models/topic_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Models and vectorizer saved successfully!")
