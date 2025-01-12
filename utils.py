import os
import re

from sentence_transformers import SentenceTransformer
import pinecone
import openai
import streamlit as st


os.environ['PINECONE_API_KEY'] = st.secrets["general"]["PINECONE_API_KEY"]
pc = pinecone.Pinecone(st.secrets["general"]["PINECONE_API_KEY"])
index = pc.Index(name='synchrotrons-index', host='https://synchrotrons-index-qdsnzxg.svc.aped-4627-b74a.pinecone.io')


def format_math_expressions(text):
    # Убираем квадратные скобки и обрамляем содержимое в $ (для выражений на нескольких строках)
    text = re.sub(r'\[\s*(.*?)\s*\]', r'$\1$', text, flags=re.DOTALL)
    text = re.sub(r'\n(?=.*\$)', '', text)
    re.sub(r'[^\w\s]*(?=\$[^\$]*$)', '', text)
    # Убираем пробелы перед и после $ внутри формулы, только если пробелы стоят до или после символов $
    text = re.sub(r'\s*\$(.*?)\s*\$', r'$\1$', text)
    text = re.sub(r'\s+\$(?=\S)', '$', text)
    text = re.sub(r'\s+\$(?=\S)', '$', text)
    # Убираем пробел перед последним знаком $ в строке
    text = re.sub(r'\s+(?=\$[^\$]*$)', '', text)
    # Обрабатываем выражения в круглых скобках () и убираем скобки, обрамляя знаком $
    text = re.sub(r'\\\((.*?)\\\)', r'$\1$', text)
    return text

