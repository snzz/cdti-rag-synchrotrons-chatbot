import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import firebase_admin
import streamlit as st
from firebase_admin import credentials, firestore

# fs_cred_str: str = st.secrets["general"]["FIRESTORE_CREDENTIALS"]
#
# # Доп обработка json credentials
# fs_cred_str = re.sub(r"'", '"', fs_cred_str)
# fs_cred_str = re.sub(r'[\f\n\r\t\v]', '', fs_cred_str)
#
# cred_json_file_path = Path('credentials.json')
#
# if not cred_json_file_path.exists():
#     with open(cred_json_file_path, 'w') as f:
#         f.write(fs_cred_str)

try:
    firebase_admin.get_app('cdti-rag-app')
except Exception as exc:
    fs_cert = credentials.Certificate('cdti-rag-synchrotrons-firebase-adminsdk-7jfem-dff3d2e7ce.json')
    firebase_admin.initialize_app(fs_cert, name='cdti-rag-app')


def update_user_profile(user_id, profile_name, message_history) -> bool:
    fs = firestore.client()
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


def get_user_profiles(user_id) -> [Any]:
    fs = firestore.client()
    # Получаем ссылку на коллекцию 'profiles' конкретного пользователя
    user_ref = fs.collection("users").document(user_id)
    profiles_ref = user_ref.collection("profiles")

    user_ref = fs.collection('users').add({
        'name': 'John Doe',
        'age': 30
    })

    # Получаем все документы в подколлекции 'profiles'
    profiles = profiles_ref.stream()

    # Проверка, есть ли профили
    profiles_list = []
    for profile in profiles:
        profile_data = profile.to_dict()  # Преобразуем документ в словарь
        profiles_list.append(profile_data)

    return profiles_list
