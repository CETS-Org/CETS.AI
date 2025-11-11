from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
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
    title="Fine-tuned TinyLlama IELTS API",
    version="2.0.0",
    description="Generate IELTS content using fine-tuned TinyLlama model."
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# GENERATOR CLASS
# ============================================================

class TinyLlamaGenerator:
    def __init__(self, model_path="./merged_model"):
        """
        Initialize merged fine-tuned TinyLlama model
        
        Args:
            model_path: Path to merged fine-tuned TinyLlama model
        """
        logger.info(f"Loading merged fine-tuned model from: {model_path}")
        
        # Check VRAM if available
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"📊 Detected VRAM: {vram_gb:.2f} GB")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Load merged model in FP16 for GPU, FP32 for CPU
        logger.info("🚀 Loading merged fine-tuned TinyLlama model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info(f"✅ Merged fine-tuned TinyLlama model loaded on {self.device}")
        if torch.cuda.is_available():
            logger.info(f"💾 VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def generate_text(self, prompt: str, max_new_tokens: int = 1200, temperature: float = 0.7) -> str:
        logger.info(f"Generating text for prompt: {prompt[:80]}...")
        
        # Prepare inputs
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2000)
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=1.15,
            )

        text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the original prompt from the generated text
        result = text.replace(prompt, "").strip()
        
        logger.info("Text generation complete.")
        return result

# ============================================================
# API ROUTES
# ============================================================

generator = None
MODEL_PATH = "./merged_model"

def initialize_model():
    global generator
    logger.info("🚀 Initializing merged fine-tuned TinyLlama model...")
    generator = TinyLlamaGenerator(MODEL_PATH)
    logger.info("✅ Model initialization complete!")

class GenerateRequest(BaseModel):
    prompt: str
    max_new_tokens: Optional[int] = 1200
    temperature: Optional[float] = 0.7

class GenerateResponse(BaseModel):
    success: bool
    prompt: str
    generated_text: str

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str

@app.on_event("startup")
async def startup_event():
    initialize_model()

@app.get("/")
def home():
    return {
        "message": "Fine-tuned TinyLlama IELTS API",
        "model": "TinyLlama-1.1B (Merged Fine-tuned for IELTS)",
        "model_path": MODEL_PATH,
        "version": "2.0.0",
        "description": "Generate IELTS reading passages and questions using merged fine-tuned TinyLlama model.",
        "endpoints": {
            "/generate": "POST - Generate IELTS content",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation"
        },
        "usage": {
            "prompt": "Your prompt for IELTS content generation",
            "max_new_tokens": "Number of tokens to generate (default: 1200)",
            "temperature": "Creativity level 0.1-2.0 (default: 0.7)"
        }
    }

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="healthy",
        model_loaded=generator is not None,
        device=str(generator.device) if generator else "N/A"
    )

@app.post("/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    if generator is None:
        raise HTTPException(500, "Model not initialized")
    if not req.prompt.strip():
        raise HTTPException(400, "Prompt must not be empty")

    try:
        text = generator.generate_text(
            req.prompt.strip(),
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature
        )
        return GenerateResponse(
            success=len(text) > 0,
            prompt=req.prompt,
            generated_text=text
        )
    except Exception as e:
        logger.error(f"Generation error: {e}")
        raise HTTPException(500, str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=5000)
