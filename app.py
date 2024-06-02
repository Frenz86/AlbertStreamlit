import onnxruntime
import numpy as np
import streamlit as st

#################################################################################################
# from huggingface_hub import hf_hub_download
# model_path = hf_hub_download(repo_id="Frenz/modelsent_test", filename="sentiment-int8.onnx")
# tokenizer_path = hf_hub_download(repo_id="Frenz/modelsent_test", filename="tokenizer_sentiment.pkl")

# import joblib
# tokenizer = joblib.load('tokenizer_sentiment.pkl')
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

@st.cache_resource  # 👈 Add the caching decorator
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("Frenz/modelsent_test")
    model = AutoModelForSequenceClassification.from_pretrained("Frenz/modelsent_test")
    return model, tokenizer


def predict_sentiment(model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    labels = ["negative", "positive"]
    predicted_label = labels[predicted_class]
    return predicted_label

def main():
    st.title("Sentiment Analysis (ALBERT) from HuggingFace 🤗 Repository")
    input_text = st.text_area("Enter text for classification", height=150)
    model, tokenizer = load_model()
    
    if st.button("Analyze"):
        if input_text:
            input_texts = input_text.split('\n')
            pred = [predict_sentiment(model, tokenizer, text) for text in input_texts][0]

            st.info(pred, icon="📑")
            st.balloons()
        else:
            st.write("Please enter text to analyze.")

if __name__ == "__main__":
    main()