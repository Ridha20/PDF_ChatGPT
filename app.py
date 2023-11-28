import streamlit as st
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import Replicate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import TextLoader
from langchain.document_loaders import Docx2txtLoader
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import os
from dotenv import load_dotenv
import tempfile


load_dotenv()
with st.sidebar:
    st.title('PDF ChatBot:books:')
    st.markdown('''
    ## About
    This application features a chatbot powered by the llama-2-70b model (LLM) , offering a user-friendly interface 
    for interactive conversations and information retrieval from uploaded PDF documents.It is built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [llama2](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/tree/main) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made with by [Ridha Isha](https://www.linkedin.com/in/ridha-isha/)')

    
def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

def conversation_chat(query, chain, history):
    result = chain({"question": query, "chat_history": history})
    history.append((query, result["answer"]))
    return result["answer"]

def display_chat_history(chain):
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your PDF", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, chain, st.session_state['history'])

                
    with reply_container:
        st.header("Chat History:")
        
        for history_item in st.session_state['history']:
            user_icon = "👨‍💼"
            bot_icon = "💬"
            st.info(f"{user_icon} User: {history_item[0]}")
            st.success(f"{bot_icon} Bot: {history_item[1]}")

def create_conversational_chain(vector_store):
    load_dotenv()
    llm = Replicate(
        streaming = True,
        model = "replicate/llama-2-70b-chat:58d078176e02c219e11eb4da5a02a7830a283b14cf8f94537af893ccff5ee781", 
        callbacks=[StreamingStdOutCallbackHandler()],
        model_kwargs = {"temperature": 0.01, "max_length" :500,"top_p":1})
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    chain = ConversationalRetrievalChain.from_llm(llm=llm, chain_type='stuff',
                                                 retriever=vector_store.as_retriever(search_kwargs={"k": 2}),
                                                 memory=memory)
    return chain

def main():
    load_dotenv()
    # Initialize session state
    initialize_session_state()
    st.title("💬 Chat with PDF 💬")
    # Initialize Streamlit
    uploaded_files = st.file_uploader("Upload your PDF", type="pdf", accept_multiple_files=True)


    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", 
                                           model_kwargs={'device': 'cpu'})

        # Create vector store
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)

        # Create the chain object
        chain = create_conversational_chain(vector_store) 

        
        display_chat_history(chain)

if __name__ == "__main__":
    main()