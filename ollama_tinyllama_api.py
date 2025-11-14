from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import logging
import ollama
import uvicorn

# ============================================================
# LOGGING
# ============================================================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============================================================
# FASTAPI APP
# ============================================================
app = FastAPI(
    title="IELTS Test Generator API (Ollama Edition)",
    version="1.0.0",
    description="Generate IELTS Reading Tests using TinyLlama via Ollama API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# OLLAMA GENERATOR CLASS
# ============================================================

class IELTSTestGenerator:
    def __init__(self, ollama_host: str, model_name: str):
        """
        Initialize connection to Ollama server
        """
        self.model_name = model_name
        self.client = ollama.Client(host=ollama_host)

        # Test connection to Ollama
        try:
            self.client.list()
            logger.info(f"🟢 Connected to Ollama at {ollama_host}")
        except Exception as e:
            logger.error(f"🔴 Cannot connect to Ollama at {ollama_host}: {e}")
            raise e

    def generate_ielts_test(
        self,
        topic: str,
        test_type: str = "reading",
        max_new_tokens: int = 1200,
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1
    ) -> str:

        if test_type.lower() == "reading":
            prompt_text = f"Create an IELTS reading passage about: {topic}. Include 1 passage and 5 questions."
        else:
            prompt_text = f"Generate an IELTS {test_type} test about: {topic}"

        # Chat-format prompt
        prompt = f"<|user|>\n{prompt_text}</s>\n<|assistant|>\n"

        response = self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options={
                "temperature": temperature,
                "top_p": top_p,
                "num_predict": max_new_tokens,
                "repeat_penalty": repetition_penalty
            }
        )

        return response["response"]

    def generate_batch(
        self,
        topics: List[str],
        test_type: str = "reading",
        **params
    ):
        results = []
        for topic in topics:
            text = self.generate_ielts_test(topic, test_type, **params)
            results.append(text)
        return results


# ============================================================
# INITIALIZATION
# ============================================================

generator = None

# 👉 SỬA 2 DÒNG NÀY CHO ĐÚNG PRIVATE SERVER CỦA BẠN
OLLAMA_HOST = "http://100.92.147.73:11434"
OLLAMA_MODEL = "tinyllama"

def initialize_model():
    global generator
    logger.info("🚀 Connecting to Ollama server...")
    generator = IELTSTestGenerator(ollama_host=OLLAMA_HOST, model_name=OLLAMA_MODEL)
    logger.info("✅ Model initialized (Ollama remote model)")


@app.on_event("startup")
async def startup_event():
    initialize_model()


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class GenerateTestRequest(BaseModel):
    topic: str
    test_type: Optional[str] = "reading"
    max_new_tokens: Optional[int] = 1200
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    repetition_penalty: Optional[float] = 1.1

class GenerateTestResponse(BaseModel):
    success: bool
    topic: str
    test_type: str
    generated_content: str
    length: int

class BatchGenerateRequest(BaseModel):
    topics: List[str]
    test_type: Optional[str] = "reading"
    max_new_tokens: Optional[int] = 1200
    temperature: Optional[float] = 0.7

class BatchGenerateResponse(BaseModel):
    success: bool
    count: int
    results: List[dict]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model: str
    ollama_host: str


# ============================================================
# ENDPOINTS
# ============================================================

@app.get("/")
def home():
    return {
        "title": "IELTS Test Generator API (Ollama Edition)",
        "model": OLLAMA_MODEL,
        "ollama_host": OLLAMA_HOST,
        "description": "Using Ollama server to generate IELTS reading passages",
        "example": {
            "topic": "technology",
            "endpoint": "/generate"
        }
    }

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=generator is not None,
        device="ollama",
        model=OLLAMA_MODEL,
        ollama_host=OLLAMA_HOST
    )

@app.post("/generate", response_model=GenerateTestResponse)
def generate_test(req: GenerateTestRequest):
    if generator is None:
        raise HTTPException(500, "Model not initialized")

    try:
        result = generator.generate_ielts_test(
            topic=req.topic,
            test_type=req.test_type,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty
        )

        return GenerateTestResponse(
            success=True,
            topic=req.topic,
            test_type=req.test_type,
            generated_content=result,
            length=len(result)
        )
    except Exception as e:
        raise HTTPException(500, str(e))

@app.post("/generate/batch", response_model=BatchGenerateResponse)
def generate_batch(req: BatchGenerateRequest):
    if generator is None:
        raise HTTPException(500, "Model not initialized")

    if len(req.topics) > 10:
        raise HTTPException(400, "Max 10 topics per batch")

    try:
        texts = generator.generate_batch(
            topics=req.topics,
            test_type=req.test_type,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature
        )

        results = [
            {"topic": t, "content": c, "length": len(c)}
            for t, c in zip(req.topics, texts)
        ]

        return BatchGenerateResponse(
            success=True,
            count=len(results),
            results=results
        )

    except Exception as e:
        raise HTTPException(500, str(e))


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5002)
