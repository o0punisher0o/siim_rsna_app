from pydantic import BaseModel
from typing import Dict, Optional


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    probabilities: Dict[str, float]
    gradcam: Optional[str] = None
