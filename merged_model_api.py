"""
IELTS Test Generator API - Merged Fine-tuned Model
Sử dụng merged model đã được fine-tune để sinh đề thi IELTS
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
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
    title="IELTS Test Generator API",
    version="1.0.0",
    description="Generate IELTS Reading Tests using Fine-tuned TinyLlama Model"
)

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

class IELTSTestGenerator:
    def __init__(self, model_path="./merged_model"):
        """
        Initialize merged fine-tuned TinyLlama model for IELTS test generation
        
        Args:
            model_path: Path to merged fine-tuned model directory
        """
        logger.info(f"Loading merged fine-tuned model from: {model_path}")
        
        # Check CUDA availability and VRAM
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"📊 Detected VRAM: {vram_gb:.2f} GB")
            logger.info(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        else:
            logger.warning("⚠️ CUDA not available, using CPU (will be slower)")
        
        # Load tokenizer
        logger.info("📝 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Set pad token if not set
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("Set pad_token = eos_token")
        
        # Load merged model
        logger.info("🚀 Loading merged fine-tuned model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"✅ Model loaded successfully on {self.device}")
        if torch.cuda.is_available():
            logger.info(f"💾 VRAM usage: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    def generate_ielts_test(
        self, 
        topic: str, 
        test_type: str = "reading",
        max_new_tokens: int = 1500, 
        temperature: float = 0.7,
        top_p: float = 0.95,
        repetition_penalty: float = 1.15
    ) -> str:
        """
        Generate IELTS test based on topic
        
        Args:
            topic: Subject/topic for the IELTS test (e.g., "climate change", "technology")
            test_type: Type of test (default: "reading")
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature (higher = more creative)
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Generated IELTS test content
        """
        logger.info(f"Generating IELTS {test_type} test on topic: {topic}")
        
        # Construct prompt using the format from training data
        if test_type.lower() == "reading":
            prompt_text = f"Create reading passage for IELTS on {topic}"
        else:
            prompt_text = f"Generate an IELTS {test_type} test on {topic}"
        
        # Format prompt with chat template (same as training)
        # Format: <|user|>\n{prompt}</s>\n<|assistant|>\n
        formatted_prompt = f"<|user|>\n{prompt_text}</s>\n<|assistant|>\n"
        
        logger.info(f"Formatted prompt: {formatted_prompt[:100]}...")
        
        # Tokenize input
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt", 
            truncation=True, 
            max_length=2048
        )
        input_ids = inputs["input_ids"].to(self.device)
        attention_mask = inputs["attention_mask"].to(self.device)

        # Generate text
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=50,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3,
                repetition_penalty=repetition_penalty,
            )

        # Decode the generated text
        full_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the assistant's response
        if "<|assistant|>" in full_text:
            result = full_text.split("<|assistant|>")[-1].strip()
        else:
            # If format is not found, remove the original prompt
            result = full_text.replace(formatted_prompt, "").strip()
        
        logger.info(f"✅ Generated {len(result)} characters")
        return result

    def generate_batch(
        self,
        topics: List[str],
        test_type: str = "reading",
        **generation_params
    ) -> List[str]:
        """
        Generate multiple IELTS tests for different topics
        
        Args:
            topics: List of topics
            test_type: Type of test
            **generation_params: Additional generation parameters
            
        Returns:
            List of generated tests
        """
        logger.info(f"Generating batch of {len(topics)} tests")
        results = []
        for i, topic in enumerate(topics, 1):
            logger.info(f"Processing {i}/{len(topics)}: {topic}")
            result = self.generate_ielts_test(topic, test_type, **generation_params)
            results.append(result)
        return results

# ============================================================
# GLOBAL GENERATOR INSTANCE
# ============================================================

generator = None
MODEL_PATH = "./merged_model"

def initialize_model():
    """Initialize the IELTS test generator on startup"""
    global generator
    logger.info("=" * 60)
    logger.info("🚀 Initializing IELTS Test Generator...")
    logger.info("=" * 60)
    generator = IELTSTestGenerator(MODEL_PATH)
    logger.info("=" * 60)
    logger.info("✅ Model initialization complete!")
    logger.info("=" * 60)

# ============================================================
# REQUEST/RESPONSE MODELS
# ============================================================

class GenerateTestRequest(BaseModel):
    topic: str
    test_type: Optional[str] = "reading"
    max_new_tokens: Optional[int] = 1500
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.95
    repetition_penalty: Optional[float] = 1.15

class GenerateTestResponse(BaseModel):
    success: bool
    topic: str
    test_type: str
    generated_content: str
    length: int

class BatchGenerateRequest(BaseModel):
    topics: List[str]
    test_type: Optional[str] = "reading"
    max_new_tokens: Optional[int] = 1500
    temperature: Optional[float] = 0.7

class BatchGenerateResponse(BaseModel):
    success: bool
    count: int
    results: List[dict]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    model_path: str

# ============================================================
# API ENDPOINTS
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    initialize_model()

@app.get("/")
def home():
    """Home endpoint with API information"""
    return {
        "title": "IELTS Test Generator API",
        "version": "1.0.0",
        "model": "TinyLlama-1.1B (Merged Fine-tuned for IELTS)",
        "model_path": MODEL_PATH,
        "description": "Generate high-quality IELTS reading passages and questions using fine-tuned TinyLlama",
        "training_info": {
            "method": "Fine-tuned with QLoRA on IELTS dataset",
            "epochs": 3,
            "format": "Chat format with <|user|> and <|assistant|> tags",
            "optimization": "Optimized for IELTS content generation"
        },
        "endpoints": {
            "/": "GET - This information page",
            "/health": "GET - Check API health status",
            "/generate": "POST - Generate single IELTS test",
            "/generate/batch": "POST - Generate multiple IELTS tests",
            "/docs": "GET - Interactive API documentation"
        },
        "usage": {
            "single_test": {
                "endpoint": "/generate",
                "method": "POST",
                "body": {
                    "topic": "climate change",
                    "test_type": "reading",
                    "max_new_tokens": 1500,
                    "temperature": 0.7
                }
            },
            "batch_tests": {
                "endpoint": "/generate/batch",
                "method": "POST",
                "body": {
                    "topics": ["technology", "health", "education"],
                    "test_type": "reading"
                }
            }
        },
        "parameters": {
            "topic": "Subject for IELTS test (e.g., 'technology', 'climate change')",
            "test_type": "Type of test (default: 'reading')",
            "max_new_tokens": "Max tokens to generate (default: 1500, range: 500-2000)",
            "temperature": "Creativity level (default: 0.7, range: 0.1-1.5)",
            "top_p": "Nucleus sampling (default: 0.95)",
            "repetition_penalty": "Avoid repetition (default: 1.15)"
        },
        "supported_topics": [
            "Technology", "Health", "Education", "Environment",
            "Business", "Science", "Culture", "Society",
            "Economics", "History", "Art", "Sports"
        ],
        "notes": [
            "Model is optimized for IELTS reading passage generation",
            "Higher temperature = more creative but less focused",
            "Typical generation time: 10-30 seconds per test",
            "Best results with specific, focused topics"
        ]
    }

@app.get("/health", response_model=HealthResponse)
def health():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy" if generator is not None else "unhealthy",
        model_loaded=generator is not None,
        device=str(generator.device) if generator else "N/A",
        model_path=MODEL_PATH
    )

@app.post("/generate", response_model=GenerateTestResponse)
def generate_test(req: GenerateTestRequest):
    """
    Generate a single IELTS test
    
    Args:
        req: Request containing topic and generation parameters
        
    Returns:
        Generated IELTS test content
    """
    if generator is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    if not req.topic.strip():
        raise HTTPException(status_code=400, detail="Topic must not be empty")
    
    try:
        generated_content = generator.generate_ielts_test(
            topic=req.topic.strip(),
            test_type=req.test_type,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature,
            top_p=req.top_p,
            repetition_penalty=req.repetition_penalty
        )
        
        return GenerateTestResponse(
            success=len(generated_content) > 0,
            topic=req.topic,
            test_type=req.test_type,
            generated_content=generated_content,
            length=len(generated_content)
        )
    except Exception as e:
        logger.error(f"❌ Generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/batch", response_model=BatchGenerateResponse)
def generate_batch(req: BatchGenerateRequest):
    """
    Generate multiple IELTS tests for different topics
    
    Args:
        req: Request containing list of topics
        
    Returns:
        List of generated tests
    """
    if generator is None:
        raise HTTPException(status_code=500, detail="Model not initialized")
    
    if not req.topics or len(req.topics) == 0:
        raise HTTPException(status_code=400, detail="Topics list must not be empty")
    
    if len(req.topics) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 topics per batch")
    
    try:
        generated_texts = generator.generate_batch(
            topics=req.topics,
            test_type=req.test_type,
            max_new_tokens=req.max_new_tokens,
            temperature=req.temperature
        )
        
        results = [
            {
                "topic": topic,
                "content": content,
                "length": len(content)
            }
            for topic, content in zip(req.topics, generated_texts)
        ]
        
        return BatchGenerateResponse(
            success=True,
            count=len(results),
            results=results
        )
    except Exception as e:
        logger.error(f"❌ Batch generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=5002,
        log_level="info"
    )

