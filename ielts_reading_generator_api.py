class IELTSReadingGenerator:
    def __init__(self, model_path="./models/phi-2", use_8bit=None, force_cpu=False):
        logger.info(f"Loading Phi-2 model from: {model_path}")
        
        # Auto-detect VRAM and decide mode
        if use_8bit is None and not force_cpu:
            if torch.cuda.is_available():
                vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
                use_8bit = vram_gb < 5
                logger.info(f"📊 Detected VRAM: {vram_gb:.2f} GB")
                if use_8bit:
                    logger.info(f"🔧 Auto-enabling 8-bit mode for low VRAM ({vram_gb:.2f} GB < 5 GB)")
                else:
                    logger.info(f"✅ Sufficient VRAM ({vram_gb:.2f} GB), using FP16 mode")
            else:
                use_8bit = False
        
        self.use_8bit = use_8bit
        self.force_cpu = force_cpu
        
        if force_cpu:
            logger.info("🔧 Force CPU mode enabled")
        elif use_8bit:
            logger.info("🔧 8-bit quantization enabled (VRAM: ~3GB)")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        try:
            # FORCE CPU MODE
            if self.force_cpu:
                logger.info("Loading model to CPU...")
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    trust_remote_code=True,
                    dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                self.device = torch.device("cpu")
                logger.info("✅ Model loaded on CPU")
                return
            
            # LOW VRAM MODE: Use 8-bit quantization
            if self.use_8bit and torch.cuda.is_available():
                logger.info("🔧 Loading model in 8-bit mode for low VRAM...")
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        load_in_8bit=True,
                        device_map="auto",
                        low_cpu_mem_usage=True
                    )
                    self.device = torch.device("cuda")
                    logger.info("✅ Model loaded in 8-bit mode")
                    vram_used = torch.cuda.memory_allocated() / 1024**3
                    logger.info(f"💾 VRAM usage: {vram_used:.2f} GB")
                    return
                except ImportError:
                    logger.warning("⚠️  bitsandbytes not installed for 8-bit mode")
                    logger.info("📝 Falling back to FP16 mode (requires more VRAM)")
                except Exception as e:
                    logger.warning(f"⚠️  8-bit loading failed: {str(e)[:100]}")
                    logger.info("📝 Falling back to FP16 mode")
            
            # NORMAL MODE
            logger.info("Attempting to load with device_map='auto'...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                trust_remote_code=True,
                device_map="auto",
                dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                low_cpu_mem_usage=True
            )
            self.device = self.model.device
            logger.info("✅ Model loaded with device_map='auto'")
        except (ValueError, ImportError) as e:
            error_msg = str(e)
            
            if "accelerate" in error_msg:
                logger.warning("⚠️  accelerate not found, loading without device_map")
            elif "disk" in error_msg or "offload" in error_msg:
                logger.warning("⚠️  Not enough VRAM/RAM for device_map='auto', loading manually")
            else:
                logger.warning(f"⚠️  Loading with device_map failed: {error_msg[:100]}")
            
            logger.info("Loading model without device_map...")
            
            try:
                if torch.cuda.is_available():
                    logger.info("Attempting to load to GPU...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        dtype=torch.float16,
                        low_cpu_mem_usage=True
                    )
                    self.device = torch.device("cuda")
                    self.model.to(self.device)
                else:
                    logger.info("CUDA not available, loading to CPU...")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.device = torch.device("cpu")
                    
                logger.info("✅ Model loaded successfully")
                
            except RuntimeError as gpu_error:
                if "out of memory" in str(gpu_error).lower():
                    logger.warning("⚠️  GPU out of memory, falling back to CPU")
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_path,
                        trust_remote_code=True,
                        dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    self.device = torch.device("cpu")
                    logger.info("✅ Model loaded on CPU")
                else:
                    raise
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        logger.info(f"✅ Phi-2 model ready on {self.device}")
        if torch.cuda.is_available():
            logger.info(f"💾 VRAM allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
