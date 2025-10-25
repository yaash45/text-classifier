from pydantic import BaseModel


class UserInputText(BaseModel):
    text: str
