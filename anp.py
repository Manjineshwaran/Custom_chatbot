import torch
import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.llms import Ollama
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from flask import Flask, render_template, request
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string

#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')

def clean_text(text):
    tokens = word_tokenize(text)

    tokens_without_punctuation = [
        word for word in tokens if word not in string.punctuation
    ]

    stop_words = set(stopwords.words('english'))
    tokens_without_stopwords = [
        word for word in tokens_without_punctuation if word.lower() not in stop_words
    ]

    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [
        lemmatizer.lemmatize(word) for word in tokens_without_stopwords
    ]

    cleaned_text = ' '.join(lemmatized_tokens)
    return cleaned_text

device = torch.device("cpu")

llm_answer_gen = Ollama(model="phi3")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MPNet-base-v2",
    model_kwargs={"device": device}
)

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    answer = ""
    if request.method == "POST":
        url = request.form.get("url")
        question = request.form.get("question")

        if url:
            try:
                response = requests.get(url)
                if response.status_code == 200:
                    soup = BeautifulSoup(response.content, "html.parser")
                    content = []
                    for element in soup.find_all():
                        text = element.get_text(strip=True)
                        if text:
                            cleaned_text = clean_text(text)
                            content.append(cleaned_text)

                    scraped_content = " ".join(content)

                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=3000, chunk_overlap=200
                    )

                    text_chunks = text_splitter.split_text(scraped_content)

                    vector_store = Chroma.from_texts(text_chunks, embeddings)

                    retriever = vector_store.as_retriever(
                        search_kwargs={"k": 5}
                    )
                    memory = ConversationBufferMemory(
                        memory_key="chat_history",
                        return_messages=True
                    )

                    answer_gen_chain = ConversationalRetrievalChain.from_llm(
                        llm=llm_answer_gen,
                        retriever=retriever,
                        memory=memory
                    )

                    if question:
                        refined_question = clean_text(question)
                        answer = answer_gen_chain.run({"question": refined_question})

                        # Ensure the answer does not include suggestions
                        if not answer or "I don't know" in answer:
                            answer = "The system could not find a relevant answer to your question."
                else:
                    answer = "Failed to fetch the webpage. Check the URL."
            except Exception as e:
                answer = f"Error occurred: {str(e)}"
        else:
            answer = "Please enter a URL."

    return render_template("index.html", answer=answer)

if __name__ == "__main__":
    app.run(debug=True)
