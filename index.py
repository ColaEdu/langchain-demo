from flask import Flask, request, jsonify
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ElasticVectorSearch, Pinecone, Weaviate, FAISS
import os

app = Flask(__name__)

# Get your API keys from openai, you will need to create an account. 
# Here is the link to get the keys: https://platform.openai.com/account/billing/overview
os.environ["OPENAI_API_KEY"] = "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"
print('reader1')
# location of the pdf file/files. 
reader = PdfReader('./LC_and_LLMI.pdf')

# read data from the file and put them into a variable called raw_text
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text
# We need to split the text that we read into smaller chunks so that during information retreival we don't hit the token size limits. 

text_splitter = CharacterTextSplitter(        
    separator = "\n",
    chunk_size = 1000,
    chunk_overlap  = 200,
    length_function = len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)
docsearch
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
@app.route('/ask_question', methods=['POST'])
def ask_question():
    # 获取用户提交的问题
    data = request.get_json()
    query = data['question']
    
    # 在这里添加您的代码，处理PDF文件并生成chain.run的结果
    # ...
    
    # 将chain.run的结果作为响应返回
    chain = load_qa_chain(OpenAI(), chain_type="stuff")
    docs = docsearch.similarity_search(query)
    result = chain.run(input_documents=docs, question=query)
    return jsonify({"answer": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

