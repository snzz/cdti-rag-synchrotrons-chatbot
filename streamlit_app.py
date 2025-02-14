import os
import uuid

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
import translators as ts

import sqlite
import utils
from utils import *


def on_add_profile_btn_click():
    st.session_state.update()
    if st.session_state['upd_prof_name'] == '':
        st.error('Название профиля не может быть пустым')
        return

    curr_user_ = st.session_state['curr_user']
    if len(curr_user_.profiles) > 10:
        st.error('Превышен лимит количества профилей на пользователя: 10')
        return

    for profile_ in curr_user_.profiles:
        if profile_.name == st.session_state['upd_prof_name']:
            st.error('Профиль с таким названием уже существует')
            return

    global system_msg
    curr_user_.profiles.append(sqlite.Profile(id=uuid.uuid4(), name=st.session_state['upd_prof_name'], history=[],
                                              responses=["Чем я могу Вам помочь?"], requests=[],
                                              prompt=system_msg))
    sqlite.update_user(user=curr_user_)
    st.session_state['curr_user'] = curr_user_
    st.session_state['selected_profile_index'] += 1
    st.session_state.update()
    st.success(f"Профиль '{st.session_state['upd_prof_name']}' был успешно добавлен")


def on_delete_profile_btn_click():
    st.session_state.update()
    curr_user_ = st.session_state['curr_user']
    selected_profile_name_ = st.session_state["selected_profile_name"]

    if len(curr_user_.profiles) <= 1:
        st.error('Минимальное количество профилей: 1')
        return

    # with st.form("ProfileDeletingForm"):
    #     st.write(f"Вы уверены, что хотите удалить профиль '{selected_profile_name_}'?")
    #     form_button_col1, form_button_col2 = st.columns(2)
    #     submitted = form_button_col1.form_submit_button("Удалить", use_container_width=True)
    #     cancelled = form_button_col2.form_submit_button("Отмена", use_container_width=True)
    #     if submitted:
    for i_, profile_ in enumerate(curr_user_.profiles):
        if profile_.name == selected_profile_name_:
            del curr_user_.profiles[i_]
            sqlite.update_user(curr_user_)
            st.session_state['curr_user'] = curr_user_
            break

    st.session_state['selected_profile_index'] = len(curr_user_.profiles) - 1
    st.session_state.update()
    st.success(f"Профиль '{st.session_state["selected_profile_name"]}' был успешно удален")


def on_change_profile_name_btn_click():
    st.session_state.update()
    curr_user_ = st.session_state['curr_user']
    selected_profile_name_ = st.session_state["selected_profile_name"]

    new_profile_name = st.session_state["upd_prof_name"]
    if new_profile_name == '':
        st.error('Название профиля не может быть пустым')
        return

    for profile_ in curr_user_.profiles:
        if profile_.name == selected_profile_name_:
            profile_.name = new_profile_name
            sqlite.update_user(curr_user_)
            st.session_state['curr_user'] = curr_user_
            st.session_state.update()
            break


def on_clear_message_history_btn_click():
    st.session_state.update()
    curr_user_ = st.session_state['curr_user']
    selected_profile_name_ = st.session_state["selected_profile_name"]

    # with st.form("ProfileMessageHistoryClearingForm"):
    #     st.write(f"Вы уверены, что хотите очистить историю сообщений профиля '{selected_profile_name_}'?")
    #     form_button_col1, form_button_col2 = st.columns(2)
    #     submitted = form_button_col1.form_submit_button("Удалить", use_container_width=True)
    #     cancelled = form_button_col2.form_submit_button("Отмена", use_container_width=True)
    #     if submitted:
    for profile_ in curr_user_.profiles:
        if profile_.name == selected_profile_name_:
            profile_.history = []
            profile_.responses = ["Чем я могу Вам помочь?"]
            profile_.requests = []
            st.session_state["history"] = profile_.history
            st.session_state['responses'] = profile_.responses
            st.session_state['requests'] = profile_.requests
            sqlite.update_user(user=curr_user_)
            st.session_state['curr_user'] = curr_user_
            st.session_state.update()
            break


st.subheader("Ассистент по теме 'Синхротроны'")

if 'responses' not in st.session_state:
    st.session_state['responses'] = ["Чем я могу Вам помочь?"]

if 'requests' not in st.session_state:
    st.session_state['requests'] = []

### Other Stuff


### НАСТРОЙКА LLM
os.environ['OPENAI_API_KEY'] = st.secrets["general"]["OPENAI_API_KEY"]
index_name = 'synchrotrons-index'
llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

if 'buffer_memory' not in st.session_state:
    st.session_state.buffer_memory = ConversationBufferWindowMemory(k=3, return_messages=True)

# Дефолтный промпт для ассистента
system_msg = ("Ты ассистент физических наук, отвечай настолько, насколько возможно правдиво, " +
              "исходя из текущего контекста. Контекст: {context}")
system_msg_template = SystemMessagePromptTemplate.from_template(template=system_msg)

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

# db
# sqlite.clear_table(sqlite.db_name, 'Users')
sqlite.init_users_table()
#

curr_user_email = st.experimental_user.email
users_collection = sqlite.get_users()
if not users_collection:
    users_collection = []
st.session_state['users'] = users_collection

curr_user: sqlite.User | None = None
for user in users_collection:
    if user.email == curr_user_email:
        curr_user = user
        break

if not curr_user:
    curr_user = sqlite.add_user(email=curr_user_email, profiles=[])

if len(curr_user.profiles) == 0:
    curr_user.profiles.append(sqlite.Profile(id=uuid.uuid4(), name='Новый профиль', history=[],
                                             responses=["Чем я могу Вам помочь?"], requests=[],
                                             prompt=system_msg))
    sqlite.update_user(user=curr_user)

st.session_state['curr_user'] = curr_user
user_profiles_cb_values = list(map(lambda p: p.name, curr_user.profiles))
if 'selected_profile_index' not in st.session_state:
    st.session_state['selected_profile_index'] = 0

st.session_state.update()
st.selectbox(label='Выберите профиль:', options=user_profiles_cb_values, key='selected_profile_name',
             index=st.session_state['selected_profile_index'])

for i, profile in enumerate(curr_user.profiles):
    if profile.name == st.session_state["selected_profile_name"]:
        st.session_state["history"] = profile.history
        st.session_state['responses'] = profile.responses
        st.session_state['requests'] = profile.requests
        st.session_state['prompt'] = profile.prompt
        # st.session_state['selected_profile_index'] = i
        break

if st.session_state['prompt'] == '':
    st.session_state['prompt'] = system_msg_template

st.text_input('Введите название профиля:', key="upd_prof_name")
st.session_state.update()

prof_name_col1, prof_name_col2 = st.columns(2)
prof_name_col1.button(label='Добавить новый профиль', use_container_width=True,
                      icon='📃', on_click=on_add_profile_btn_click)
prof_name_col2.button(label='Изменить название текущего профиля', use_container_width=True,
                      icon='✍🏻', on_click=on_change_profile_name_btn_click)

with st.expander('Удаление данных профиля', icon='❗'):
    st.button(label='Удалить выбранный профиль', use_container_width=True, icon='❌',
              on_click=on_delete_profile_btn_click, disabled=len(curr_user.profiles) == 0)
    st.button(label='Очистить историю сообщений', use_container_width=True, icon='🧹',
              on_click=on_clear_message_history_btn_click, disabled=len(curr_user.profiles) == 0)

with st.expander('Параметры чата', icon='🔧'):
    # Выбор элемента в ComboBox
    st.session_state.update()
    default_prompt_str = st.text_area('Стандартный промпт ассистента:', value=st.session_state['prompt'])
    
    if st.button('Сохранить', use_container_width=True):
        if 'Контекст: {context}' not in default_prompt_str:
            default_prompt_str += ' Контекст: {context}'

        st.session_state['prompt'] = default_prompt_str
        prompt_template = ChatPromptTemplate.from_messages(
            [default_prompt_str, MessagesPlaceholder(variable_name="history"), human_msg_template]
        )
        qa.combine_docs_chain.llm_chain.prompt = prompt_template

        for i, profile in enumerate(curr_user.profiles):
            if profile.name == st.session_state["selected_profile_name"]:
                curr_user.profiles[i] = sqlite.Profile(id=profile.id, name=profile.name,
                                                       history=st.session_state["history"],
                                                       responses=st.session_state["responses"],
                                                       requests=st.session_state["requests"],
                                                       prompt=st.session_state['prompt'])
                sqlite.update_user(curr_user)
                break

st.subheader('Чат')
# container for chat history
response_container = st.container()
# container for text box
textcontainer = st.container()

with textcontainer:
    query = st.text_input("Запрос: ", key="input", placeholder='Введите запрос')
    if query.strip():
        with st.spinner("Печатает..."):
            formatted_history = [
                HumanMessage(content=msg[0]) if i % 2 == 0 else AIMessage(content=msg[1])
                for i, msg in enumerate(st.session_state["history"])
            ]

            # Добавление английской версии запроса для получение информации из англоязычных источников
            processed_query = f' {ts.translate_text(query_text=query, translator='google', to_language='en')}'

            response = qa(
                {"question": processed_query, "history": formatted_history, "chat_history": formatted_history}
            )

            answer = utils.format_math_expressions(response["answer"])
            # Перевод ответа на русский
            answer = ts.translate_text(query_text=answer, translator='google', to_language='ru')

            # Сохранение вопроса и ответа в контексте
            st.session_state["history"].append((processed_query, answer))

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
        for profile in curr_user.profiles:
            if profile.name == st.session_state["selected_profile_name"]:
                profile = sqlite.Profile(id=profile.id, name=profile.name,
                                         history=st.session_state["history"],
                                         responses=st.session_state["responses"],
                                         requests=st.session_state["requests"],
                                         prompt=profile.prompt)
                sqlite.update_user(curr_user)
                break

with response_container:
    if st.session_state['responses']:
        for i in range(len(st.session_state['responses'])):
            with st.chat_message(name='ai', avatar='nic_ki.svg'):
                st.write(st.session_state['responses'][i])
                if i < len(st.session_state['requests']):
                    with st.chat_message(name='user', avatar='rosatom.png', is_user=True):
                        st.write(st.session_state["requests"][i])
