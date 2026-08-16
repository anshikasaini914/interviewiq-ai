from pydantic import BaseModel

class InterviewState(BaseModel):
    phase: str = "greeting"          # "greeting" | "technical" | "wrapup"
    question_count: int = 0
    max_questions: int = 4
    correct_count: int = 0
    incorrect_count: int = 0