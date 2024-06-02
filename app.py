from huggingface_hub import hf_hub_download
import onnxruntime
import joblib
import numpy as np
import streamlit as st

#################################################################################################

model_path = hf_hub_download(repo_id="Frenz/modelsent_test", filename="sentiment-int8.onnx")
tokenizer_path = hf_hub_download(repo_id="Frenz/modelsent_test", filename="tokenizer_sentiment.pkl")

tokenizer = joblib.load(tokenizer_path)                 # load tokenizer
onnx_model_path = model_path                            # load model quantized int8
ort_session = onnxruntime.InferenceSession(onnx_model_path)

def analyze_sentimentinference(text, ort_session, tokenizer):
    inputs = tokenizer(text, return_tensors="pt")
    ort_inputs = {k: v.cpu().numpy() for k, v in inputs.items()}
    ort_outs = ort_session.run(None, ort_inputs)
    probabilities = np.exp(ort_outs[0][0]) / np.exp(ort_outs[0][0]).sum(-1, keepdims=True) # prob
    sentiment = "Positive" if probabilities[1] > probabilities[0] else "Negative"
    return sentiment, probabilities[1], probabilities[0]


def main():

    st.title("Sentiment Analysis (ALBERT) from HuggingFace 🤗 Repository")
    input_text = st.text_area("Enter text for classification", height=150)
    if st.button("Analyze"):
        if input_text:
            input_texts = input_text.split('\n')
            pred,pp,ps = analyze_sentimentinference(input_texts, ort_session, tokenizer)

            st.info(pred, icon="📑")
            st.balloons()
        else:
            st.write("Please enter text to analyze.")


if __name__ == "__main__":
    main()