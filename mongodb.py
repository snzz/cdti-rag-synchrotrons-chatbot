import ssl
import pymongo
import certifi
import streamlit as st
import uuid


def get_users_collection():
    ca = certifi.where()
    connection_string = st.secrets["general"]["MONGODB_ATLAS_CONNECTION_STRING"]
    client = pymongo.MongoClient(connection_string, tlsCAFile=ca, ssl_cert_reqs=ssl.CERT_NONE)
    db = client['streamlit-chat-bot-db']
    collection = db['users']
    return collection


def insert_data(collection, data):
    collection.insert_one(data)


def read_data(collection):
    data = collection.find()
    return list(data)


def get_user_data_by_email(collection, user_email):
    query = {"email": user_email}
    user_data = collection.find_one(query)
    if user_data:
        return user_data
    else:
        return None


def add_user(collection, user_email):
    user = {
        "id": str(uuid.uuid4()),
        "email": user_email,
        "profiles": [
            {
                "id": str(uuid.uuid4()),
                "name": "Новый профиль",
                "messages": []
            },
        ],
    }
    collection.insert_one(user)
    return user


def add_user_profile(collection, user_email, profile_name, profile_messages):
    pass


def delete_user_profile(collection, user_email, profile_id):
    pass


def update_user_profile(collection, user_email, profile_id):
    pass


def update_data(collection, name, new_age):
    query = {"name": name}
    new_values = {"$set": {"age": new_age}}
    collection.update_one(query, new_values)


def delete_data(collection, name):
    query = {"name": name}
    collection.delete_one(query)

