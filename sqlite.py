import sqlite3
import pickle
import uuid
from dataclasses import dataclass
from typing import Any


db_name = 'cdti_rag_db.db'


@dataclass
class ProfileInfo:
    id: uuid.UUID
    name: str
    messages: [str]


@dataclass
class User:
    id: int | None
    email: str
    profiles: [Any]


def connect_to_db() -> Any:
    conn = sqlite3.connect('example.db')
    return conn


def init_users_table():
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS Users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT NOT NULL,
        profiles BLOB
    )
    ''')
    conn.commit()
    conn.close()


def add_user(email: str, profiles: [ProfileInfo]) -> User:
    serialized_data = pickle.dumps(profiles)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''
    INSERT INTO Users (email, profiles) VALUES (?, ?)
    ''', (email, serialized_data))
    conn.commit()
    user_id = cursor.lastrowid
    conn.close()
    return User(id=user_id, email=email, profiles=profiles)


def get_user(user_id: int) -> User | None:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM Users WHERE id = ?''', (user_id,))
    row = cursor.fetchone()
    conn.close()

    if row:
        user_id = row['id']
        user_email = row['email']
        serialized_data = row['profiles']
        profiles = pickle.loads(serialized_data)
        user = User(id=user_id, email=user_email, profiles=profiles)
        return user
    else:
        return None


def get_users() -> [User]:
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute('''SELECT * FROM Users''')
    rows = cursor.fetchall()
    conn.close()

    users = []
    for row in rows:
        user = User(id=row['id'], email=row['email'], profiles=row['profiles'])
        users.append(user)
    return users


def update_user_profiles(user_id: int, profiles: [ProfileInfo]):
    user = get_user(user_id)
    if user:
        user.profiles = profiles
        serialized_data = pickle.dumps(profiles)

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute('''
        UPDATE Users SET profiles = ? WHERE id = ?
        ''', (serialized_data, user_id))
        conn.commit()
        conn.close()
    else:
        raise ValueError(f"User with id {user_id} not found")

