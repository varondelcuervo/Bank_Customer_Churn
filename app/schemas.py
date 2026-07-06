from pydantic import BaseModel

class Customer(BaseModel):

    CreditScore: int
    Geography: str
    Gender: str
    Age: float
    Tenure: int
    Balance: float
    NumOfProducts: int
    HasCrCard: int
    IsActiveMember: int
    EstimatedSalary: float