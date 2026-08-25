import hashlib
import bcrypt

def TextEncoder(text:str):
    hashed = hashlib.sha256(text.encode()).hexdigest()
    return hashed
def PasswordEncoder(password:str):
    password_bytes = password.encode()
    hashed = bcrypt.hashpw(password_bytes,bcrypt.gensalt())
    return hashed

def ComparePasswords(login_password:str,stored_hashed_password:bytes)-> bool:
    return bcrypt.checkpw(login_password.encode(),stored_hashed_password)

def uploaded_file_encoder(uploaded_file):
    file_bytes = uploaded_file.getvalue()
    return hashlib.sha256(file_bytes).hexdigest()