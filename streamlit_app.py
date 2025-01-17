import os

from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage
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
from firestore import *


st.subheader("Ассистент по теме 'Синхротроны'")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Чем я могу Вам помочь?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

### НАСТРОЙКА LLM
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
### НАСТРОЙКА LLM

curr_user_email = st.experimental_user.email
if curr_user_email == 'ikulakov013@gmail.com':
    st.success('test')

with st.expander("Параметры чата"):
    # Исходный список значений для ComboBox
    options = ['Option 1', 'Option 2', 'Option 3']

    # Выбор элемента в ComboBox
    selected_option = st.selectbox('Choose an option to rename:', options)

    # Ввод нового имени для выбранного элемента
    new_name = st.text_input('Enter new name for the selected option:', value=selected_option)

    # Если пользователь изменяет название
    if new_name != selected_option:
        # Обновляем список, заменяя старое значение на новое
        options[options.index(selected_option)] = new_name

    # Отображаем обновленный ComboBox с новым списком
    selected_option = st.selectbox('Choose an option:', options)

    st.text_area('Стандартный промпт ассистента')

st.subheader('Чат')
# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

if "history" not in st.session_state:
    st.session_state["history"] = []

with textcontainer:
    query = st.text_input("Запрос: ", key="input", placeholder='Введите запрос')
    if query.strip():
        with st.spinner("Печатает..."):
            formatted_history = [
                HumanMessage(content=msg[0]) if i % 2 == 0 else AIMessage(content=msg[1])
                for i, msg in enumerate(st.session_state["history"])
            ]
            response = qa(
                {"question": query, "history": formatted_history, "chat_history": formatted_history}
            )

            # # debug
            # with st.chat_message('assistant'):
            #     st.write(response["answer"])
            # #

            answer = utils.format_math_expressions(response["answer"])
            # Сохранение вопроса и ответа в контексте
            st.session_state["history"].append((query, answer))

            # Отображение источников
            source_docs = []
            answer += '\n\n**Источники:**'
            for doc in response["source_documents"]:
                doc_str = f'\n- {doc.metadata.get('source', 'Неизвестный источник')}'
                if doc_str not in source_docs:
                    source_docs.append(doc_str)
                    answer += doc_str

        st.session_state.requests.append(query)
        st.session_state.responses.append(answer)

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            message(st.session_state['responses'][i], key=str(i))
            if i < len(st.session_state['requests']):
                message(st.session_state["requests"][i], is_user=True, key=str(i) + '_user')
