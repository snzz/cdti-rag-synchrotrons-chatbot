import json
import re
from dataclasses import dataclass
from pathlib import Path

import firebase_admin
import streamlit as st
from firebase_admin import credentials, firestore

fs_cred_str: str = st.secrets["general"]["FIRESTORE_CREDENTIALS"]
# Ординарные кавычки на двойные
fs_cred_str = re.sub(r"'", '"', fs_cred_str)

fs_cred_json = json.loads(fs_cred_str)
cred_json_file_path = Path('credentials.json')

if not cred_json_file_path.exists():
    with open('credentials.json', 'w') as json_file:
        json.dump(fs_cred_json, json_file, indent=4)

fs_cert = credentials.Certificate(cert='credentials.json')
firebase_admin.initialize_app(fs_cert)
fs = firestore.client()


def update_user_profile(user_id, profile_name, message_history) -> bool:
    # Создаем документ для пользователя в коллекции 'users'
    user_ref = fs.collection("users").document(user_id)
    # Добавляем подколлекцию 'profiles' для данного пользователя
    profile_ref = user_ref.collection("profiles").document(profile_name)
    profile_doc = profile_ref.get()

    if not profile_doc.exists:
        # Данные профиля
        profile_data = {
            "profile_name": profile_name,
            "message_history": message_history,
        }
        # Запись данных в Firestore
        profile_ref.set(profile_data)
        return True
    return False


def get_user_profiles(user_id):
    # Получаем ссылку на коллекцию 'profiles' конкретного пользователя
    user_ref = fs.collection("users").document(user_id)
    profiles_ref = user_ref.collection("profiles")

    # Получаем все документы в подколлекции 'profiles'
    profiles = profiles_ref.stream()

    # Проверка, есть ли профили
    profiles_list = []
    for profile in profiles:
        profile_data = profile.to_dict()  # Преобразуем документ в словарь
        profiles_list.append(profile_data)

    return profiles_list
