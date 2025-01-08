from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st


openai.api_key = st.secrets["general"]["OPENAI_API_KEY"]
model = SentenceTransformer('all-MiniLM-L6-v2')

pc = pinecone.Pinecone(
        st.secrets["general"]["PINECONE_API_KEY"]
)

index = pc.Index(host='synchrotrons-index')


def find_match(input):
    input_em = model.encode(input).tolist()
    result = index.query(vector=input_em, top_k=2, include_metadata=True)
    return result['matches'][0]['metadata']['text'] + "\n" + result['matches'][1]['metadata']['text']


def query_refiner(conversation, query):
    # response = openai.Completion.create(
    #     model="text-davinci-003",
    #     prompt=f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:",
    #     temperature=0.7,
    #     max_tokens=256,
    #     top_p=1,
    #     frequency_penalty=0,
    #     presence_penalty=0
    # )
    response = openai.chat.completions.create(
        model="gpt-4o",  # или gpt-4
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user",
             "content": f"Given the following user query and conversation log, formulate a question that would be the most relevant to provide the user with an answer from a knowledge base.\n\nCONVERSATION LOG: \n{conversation}\n\nQuery: {query}\n\nRefined Query:"},
        ],
        temperature=0
    )

    return response.choices[0].message.content


def get_conversation_string():
    conversation_string = ""
    for i in range(len(st.session_state['responses']) - 1):
        conversation_string += "Human: " + st.session_state['requests'][i] + "\n"
        conversation_string += "Bot: " + st.session_state['responses'][i + 1] + "\n"
    return conversation_string
