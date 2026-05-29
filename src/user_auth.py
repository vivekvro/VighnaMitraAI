import sqlite3
import psycopg
import re,os
from src.encrypt import PasswordEncoder,ComparePasswords
from textwrap import dedent
from dotenv import load_dotenv
load_dotenv()
DB_PATH = os.getenv('DB_POSTGRES_URL')






#------------------------ Create Table -------------------------------

#------------------------ password -------------------------------



# validate

def validate_password(pw):
    if len(pw) < 8:
        return False
    if not re.search(r"[A-Z]", pw):
        return False
    if not re.search(r"[a-z]", pw):
        return False
    if not re.search(r"\d", pw):
        return False
    if not re.search(r"[!@#$%^&*]", pw):
        return False
    return True



# fetch


def fetch_password_by_username(username:str,db_path:str=DB_PATH):
    with psycopg.connect(db_path) as con:
        cur = con.cursor()
        cur.execute( "SELECT password FROM accounts_info WHERE username = %s",(username,))
        row = cur.fetchone()
    return row[0] if row else None


#------------------------ check if Xyz exists-----------------
def check_if_user_exists(username:str,db_path:str=DB_PATH):
    with psycopg.connect(db_path) as con:
        cur = con.cursor()
        cur.execute("SELECT username from accounts_info where username = %s",(username,))
        row = cur.fetchone()
    return row is not None

def check_if_email_exists(email:str,db_path:str=DB_PATH):
    with psycopg.connect(db_path) as con:
        cur = con.cursor()
        cur.execute("SELECT email from accounts_info where email = %s",(email,))
        row = cur.fetchone()
    return row is not None


#------------------------ Signup -------------------------------

def insert_account_info(username:str,password:str,dob:str,email:str,db_path:str=DB_PATH):
    if not all([username,password,dob,email]):
        raise ValueError(dedent("All the fields are required"))
    username= username.lower().strip()
    email = email.lower().strip()
    encoded_pwd = PasswordEncoder(password=password)
    try:
        with psycopg.connect(db_path) as con:
            cur = con.cursor()
            cur.execute("""INSERT INTO accounts_info
                                (username, password, dob, email) VALUES (%s, %s, %s, %s);
                        """,(username, encoded_pwd, dob, email))
            con.commit()
    except psycopg.IntegrityError:
        raise ValueError(dedent("username or email already exists"))




#------------------------ Login-------------------------------
def login_account(username:str,password:str,db_path:str):
    if not username or  not password:
        raise ValueError(dedent("username and password required"))
    stored_pwd = fetch_password_by_username(username=username,db_path=db_path)
    if stored_pwd is None:
            raise ValueError(dedent("user does not exist, please signup first!"))
    status = ComparePasswords(login_password=password,stored_hashed_password=stored_pwd)
    return status
