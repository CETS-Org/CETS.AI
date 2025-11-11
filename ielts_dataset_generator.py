import json
import random
from typing import List, Dict

# Định nghĩa các topics và từ khóa liên quan
TOPICS = {
    "Environment": ["climate change", "pollution", "renewable energy", "deforestation", "conservation", "biodiversity", "sustainability"],
    "Technology": ["artificial intelligence", "automation", "internet", "robotics", "biotechnology", "virtual reality", "cybersecurity"],
    "Education": ["online learning", "curriculum", "teaching methods", "student performance", "educational technology", "literacy", "higher education"],
    "Health": ["mental health", "nutrition", "exercise", "medical research", "public health", "disease prevention", "healthcare systems"],
    "Culture": ["traditions", "cultural heritage", "art", "music", "language preservation", "cultural diversity", "festivals"],
    "Economy": ["globalization", "trade", "employment", "economic growth", "innovation", "entrepreneurship", "financial markets"],
    "Science": ["space exploration", "genetics", "physics", "chemistry", "marine biology", "neuroscience", "quantum computing"],
    "Society": ["urbanization", "population growth", "social media", "migration", "aging population", "inequality", "community development"],
    "History": ["ancient civilizations", "historical events", "archaeological discoveries", "industrial revolution", "colonialism", "world wars", "cultural evolution"],
    "Psychology": ["human behavior", "cognitive development", "memory", "emotions", "social psychology", "personality", "learning theories"]
}

# Template cho các loại câu hỏi IELTS Reading
QUESTION_TYPES = [
    "Multiple Choice",
    "True/False/Not Given",
    "Yes/No/Not Given",
    "Matching Headings",
    "Sentence Completion",
    "Summary Completion",
    "Matching Information",
    "Short Answer Questions"
]

# Template prompts người dùng có thể hỏi
USER_PROMPTS = [
    "Create an IELTS reading passage about {topic}",
    "Generate an IELTS reading test on {topic}",
    "I need an IELTS reading passage about {keyword}",
    "Make an IELTS reading exercise on {topic}",
    "Create a reading comprehension test about {keyword}",
    "Generate IELTS reading material on {topic}",
    "Produce an IELTS reading passage covering {keyword}",
    "Design an IELTS reading test about {topic}",
    "Create reading passage for IELTS on {keyword}",
    "Make IELTS reading questions about {topic}"
]

def generate_reading_passage(topic: str, keyword: str, word_count: int) -> str:
    """Generate một đoạn reading logic với số từ yêu cầu"""
    
    # Templates cho các phần của đoạn văn
    introductions = [
        f"In recent years, {keyword} has become an increasingly important topic in the field of {topic.lower()}. Researchers and experts have devoted considerable attention to understanding its implications and potential impact on society.",
        f"The study of {keyword} within {topic.lower()} has evolved significantly over the past few decades. This transformation has been driven by technological advances and changing societal needs.",
        f"{keyword.capitalize()} represents a crucial aspect of modern {topic.lower()}. Its influence extends across multiple domains, affecting how we live, work, and interact with the world around us.",
        f"Understanding {keyword} is essential for anyone interested in {topic.lower()}. The complexities surrounding this subject require careful examination and critical thinking."
    ]
    
    body_templates = [
        f"One of the primary considerations regarding {keyword} involves its practical applications. Studies have shown that implementing effective strategies can lead to significant improvements in outcomes. Researchers have documented numerous cases where innovative approaches have yielded positive results. Furthermore, the integration of modern methodologies has enhanced our ability to address related challenges. These developments suggest that continued investment in this area will likely produce substantial benefits for society as a whole.",
        
        f"The historical context of {keyword} provides valuable insights into current trends. Early pioneers in {topic.lower()} laid the groundwork for contemporary understanding through their groundbreaking research. Over time, theories and practices have evolved, incorporating new evidence and perspectives. This evolution reflects broader changes in scientific methodology and societal values. Modern practitioners now have access to tools and resources that were unimaginable just a generation ago, enabling more sophisticated analysis and intervention.",
        
        f"Several factors contribute to the complexity of {keyword}. Environmental conditions, economic considerations, and social dynamics all play crucial roles in shaping outcomes. Experts emphasize the importance of taking a holistic approach that considers multiple variables simultaneously. This multifaceted perspective allows for more nuanced understanding and more effective problem-solving strategies. Additionally, interdisciplinary collaboration has proven essential for advancing knowledge in this field.",
        
        f"Critics and proponents of various approaches to {keyword} have engaged in ongoing debate about best practices. Some argue that traditional methods remain most effective, while others advocate for innovative solutions. Evidence suggests that both perspectives offer valuable contributions, and the optimal approach may vary depending on specific circumstances. This diversity of viewpoints enriches the discourse and promotes continuous improvement in the field."
    ]
    
    conclusions = [
        f"Looking ahead, the future of {keyword} in {topic.lower()} appears promising yet challenging. Continued research, thoughtful policy-making, and sustained public engagement will be essential for realizing potential benefits while mitigating risks. As our understanding deepens, we can expect to see further innovations that transform how we approach these important issues.",
        
        f"In conclusion, {keyword} remains a dynamic and evolving area within {topic.lower()}. The insights gained from decades of research provide a solid foundation for future progress. However, much work remains to be done. Addressing remaining questions will require dedication, creativity, and collaboration across disciplines and sectors.",
        
        f"The implications of {keyword} for {topic.lower()} cannot be overstated. As we move forward, maintaining a balance between innovation and caution will be crucial. By learning from both successes and failures, we can develop more effective strategies that benefit individuals and communities alike."
    ]
    
    # Xây dựng đoạn văn
    passage = random.choice(introductions) + " "
    
    # Thêm các đoạn body cho đến khi đạt word count mong muốn
    current_word_count = len(passage.split())
    while current_word_count < word_count - 100:
        passage += random.choice(body_templates) + " "
        current_word_count = len(passage.split())
    
    # Thêm kết luận
    passage += random.choice(conclusions)
    
    return passage

def generate_questions(passage: str, topic: str, num_questions: int) -> List[Dict]:
    """Generate câu hỏi IELTS theo các dạng khác nhau"""
    questions = []
    
    # Lấy một số từ khóa từ đoạn văn để tạo câu hỏi
    words = passage.split()
    
    for i in range(num_questions):
        q_type = random.choice(QUESTION_TYPES)
        
        if q_type == "Multiple Choice":
            question = {
                "type": "Multiple Choice",
                "question": f"According to the passage, what is a key aspect of {topic.lower()}?",
                "options": [
                    "A) It has remained unchanged over time",
                    "B) It requires interdisciplinary collaboration",
                    "C) It is only relevant to specific regions",
                    "D) It has no practical applications"
                ],
                "answer": "B"
            }
        
        elif q_type in ["True/False/Not Given", "Yes/No/Not Given"]:
            statements = [
                f"Research in {topic.lower()} has evolved significantly in recent decades.",
                f"Traditional methods are always more effective than modern approaches.",
                f"The future developments in this field are entirely predictable.",
                f"Multiple factors contribute to the complexity of this subject."
            ]
            question = {
                "type": q_type,
                "statement": random.choice(statements),
                "answer": random.choice(["True", "False", "Not Given"]) if "True/False" in q_type else random.choice(["Yes", "No", "Not Given"])
            }
        
        elif q_type == "Sentence Completion":
            question = {
                "type": "Sentence Completion",
                "question": f"Complete the sentence: Modern practitioners now have access to _____ that enable more sophisticated analysis.",
                "answer": "tools and resources"
            }
        
        elif q_type == "Short Answer Questions":
            short_questions = [
                f"What has driven the transformation in {topic.lower()}? (Maximum 3 words)",
                "What type of approach do experts emphasize? (Maximum 2 words)",
                "What will be essential for realizing potential benefits? (Maximum 3 words)"
            ]
            question = {
                "type": "Short Answer Questions",
                "question": random.choice(short_questions),
                "answer": random.choice(["technological advances", "holistic approach", "continued research"])
            }
        
        else:  # Summary Completion hoặc các dạng khác
            question = {
                "type": q_type,
                "question": f"Which section discusses the historical development of {topic.lower()}?",
                "answer": "Paragraph 2"
            }
        
        questions.append(question)
    
    return questions

def create_training_sample(topic: str, keyword: str) -> Dict:
    """Tạo 1 sample training theo format GPT-2"""
    
    # Tạo user prompt
    prompt_template = random.choice(USER_PROMPTS)
    user_prompt = prompt_template.format(topic=topic, keyword=keyword)
    
    # Tạo reading passage
    word_count = random.randint(500, 700)
    passage = generate_reading_passage(topic, keyword, word_count)
    
    # Tạo câu hỏi
    num_questions = random.randint(10, 15)
    questions = generate_questions(passage, topic, num_questions)
    
    # Format output
    output = f"Reading Passage:\n\n{passage}\n\n"
    output += "Questions:\n\n"
    
    for idx, q in enumerate(questions, 1):
        if q["type"] == "Multiple Choice":
            output += f"{idx}. {q['question']}\n"
            for opt in q["options"]:
                output += f"   {opt}\n"
            output += f"Answer: {q['answer']}\n\n"
        elif q["type"] in ["True/False/Not Given", "Yes/No/Not Given"]:
            output += f"{idx}. {q['statement']}\n"
            output += f"Answer: {q['answer']}\n\n"
        else:
            output += f"{idx}. {q['question']}\n"
            output += f"Answer: {q['answer']}\n\n"
    
    return {
        "prompt": user_prompt,
        "completion": output
    }

def generate_dataset(num_samples: int = 7000, output_file: str = "ielts_reading_dataset.jsonl"):
    """Generate toàn bộ dataset"""
    
    print(f"Generating {num_samples} training samples...")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # Chọn topic và keyword ngẫu nhiên
            topic = random.choice(list(TOPICS.keys()))
            keyword = random.choice(TOPICS[topic])
            
            # Tạo sample
            sample = create_training_sample(topic, keyword)
            
            # Ghi vào file theo format JSONL
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
            
            if (i + 1) % 500 == 0:
                print(f"Generated {i + 1}/{num_samples} samples...")
    
    print(f"Dataset saved to {output_file}")
    print(f"Total samples: {num_samples}")

def preview_sample():
    """Xem thử 1 sample"""
    topic = "Technology"
    keyword = "artificial intelligence"
    sample = create_training_sample(topic, keyword)
    
    print("=" * 80)
    print("SAMPLE PREVIEW")
    print("=" * 80)
    print(f"\nPROMPT:\n{sample['prompt']}")
    print(f"\nCOMPLETION:\n{sample['completion'][:500]}...")
    print("=" * 80)

if __name__ == "__main__":
    # Xem thử 1 sample trước
    preview_sample()
    
    # Generate dataset
    print("\nStarting dataset generation...")
    generate_dataset(num_samples=7000, output_file="ielts_reading_dataset.jsonl")
    
    print("\n✓ Dataset generation complete!")
    print("\nFile format: JSONL (JSON Lines)")
    print("Each line contains: {'prompt': '...', 'completion': '...'}")
    print("\nYou can use this dataset to fine-tune GPT-2 or other language models.")