from pydantic import BaseModel

class InterviewState(BaseModel):
    phase: str = "greeting"           # "greeting" | "technical" | "wrapup"
    intro_done: bool = False
    question_count: int = 0
    max_questions: int = 12
    correct_count: int = 0
    incorrect_count: int = 0     
    consecutive_wrong: int = 0        
    greeting_exchange_count: int = 0
   
   