import os

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
    MessagesPlaceholder
)
import streamlit as st
from streamlit_chat import message

import utils
from utils import *

st.subheader("Ассистент по теме 'Синхротроны'")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Чем я могу Вам помочь?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

os.environ['OPENAI_API_KEY'] = st.secrets["general"]["OPENAI_API_KEY"]
index_name = 'synchrotrons-index'
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Дефолтный промпт для ассистента
system_msg_template = SystemMessagePromptTemplate.from_template(template="""
Ты ассистент физических наук, отвечай настолько, насколько возможно правдиво, исходя из текущего контекста.
Контекст: {context}
""")

human_msg_template = HumanMessagePromptTemplate.from_template(template="{question}")

prompt_template = ChatPromptTemplate.from_messages(
    [system_msg_template, MessagesPlaceholder(variable_name="history"), human_msg_template]
)

embeddings = OpenAIEmbeddings(
    model='text-embedding-3-small'
)
vectorstore = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)

qa = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True,
    verbose=True
)
qa.combine_docs_chain.llm_chain.prompt = prompt_template

# qa = RetrievalQA.from_chain_type(
#     llm=llm,
#     chain_type='stuff',
#     retriever=vectorstore.as_retriever(),
# )

# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

if "history" not in st.session_state:
    st.session_state["history"] = []

with textcontainer:
    query = st.text_input("Запрос: ", key="input", placeholder='Введите запрос')
    if st.button("Отправить"):
        if query.strip():
            with st.chat_message("assistant"):
                with st.spinner("Печатает..."):
                    response = qa(
                        {"question": query, "history": st.session_state["history"], "chat_history": st.session_state["history"]}
                    )

                    answer = response["answer"]
                    # Сохранение вопроса и ответа в контексте
                    st.session_state["history"].append((query, answer))

                    # Отображение ответа
                    st.write(f"**Ответ:** {answer}")

                    # Отображение источников
                    st.write("**Источники:**")
                    for doc in response["source_documents"]:
                        st.write(f"- {doc.metadata.get('source', 'Неизвестный источник')}")

                    # # Получаем историю диалога из памяти
                    # chat_history = st.session_state.buffer_memory.load_memory_variables({}).get('history', [])
                    # if chat_history is None:
                    #     chat_history = []
                    #
                    # # Вызываем цепочку с правильными входными данными
                    # response = qa.invoke(query)['result']
                    # response = utils.format_math_expressions(response)
                    #
                    # # Сохраняем контекст
                    # st.session_state.buffer_memory.save_context({"input": query}, {"output": response})

                st.session_state.requests.append(query)
                st.session_state.responses.append(answer)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
