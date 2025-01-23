import sqlite3
import pickle
import uuid
from dataclasses import dataclass, field
from typing import Any


db_name = 'cdti_rag_db.db'


@dataclass
class Profile:
    id: uuid.UUID
    name: str
    history: list = field(default_factory=list)
    responses: list = field(default_factory=list)
    requests: list = field(default_factory=list)
    prompt: str = ""


@dataclass
class User:
    id: int | None
    email: str
    profiles: [Profile]


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


def clear_table(db: str, table_name: str):
    try:
        conn = sqlite3.connect(db)
        cursor = conn.cursor()
        cursor.execute(f"DELETE FROM {table_name};")
        conn.commit()
    except sqlite3.Error as e:
        print(f"Ошибка при очистке таблицы: {e}")
    finally:
        if conn:
            conn.close()


def add_user(email: str, profiles: [Profile]) -> User:
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
        user_id = row[0]            # id
        user_email = row[1]         # email
        serialized_data = row[2]    # profiles
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
        user = User(id=row[0], email=row[1], profiles=pickle.loads(row[2]))
        users.append(user)
    return users


def update_user(user: User):
    if user:
        serialized_data = pickle.dumps(user.profiles)

        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute('''
        UPDATE Users SET email = ?, profiles = ? WHERE id = ?
        ''', (user.email, serialized_data, user.id))
        conn.commit()
        conn.close()
    else:
        raise ValueError(f"User with id {user.id} not found")

