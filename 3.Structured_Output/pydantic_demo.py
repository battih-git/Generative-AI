from pydantic import BaseModel, EmailStr
from typing import Optional

class Student(BaseModel):
    name: str = "Huzefa"
    age: int = None
    Occupation: Optional[str] = None


new_student = {'age': 31, 'Occupation':'Business', 'email':'huzefabattiwala.hb@gmail.com'}

student = Student(**new_student)

print(student)
print(type(Student))