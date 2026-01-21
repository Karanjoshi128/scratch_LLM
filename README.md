# 🤖 Scratch LLM: Building Large Language Models from Scratch

An educational repository implementing Large Language Models (LLMs) and GPT models from scratch, covering foundational concepts to advanced fine-tuning techniques.

---

## 📚 Project Overview

This project is a comprehensive learning guide that walks through building Large Language Models step-by-step, from understanding fundamental concepts to implementing production-ready techniques. Each chapter combines theoretical explanations with practical code implementations using PyTorch.

**Key Focus Areas:**

- Understanding LLM architecture and transformer mechanisms
- Text tokenization and data processing
- Attention mechanisms and multi-head attention
- GPT model implementation and training
- Fine-tuning strategies (standard and instruction-based)
- Model evaluation and deployment considerations

---

## 📖 Chapter Breakdown

### **Chapter 1: Understanding Large Language Models**

- Foundational concepts of Large Language Models
- Overview of the LLM development lifecycle covered in this book
- Setup guidance for Python environments
- Video tutorials for getting started

**Resources:**

- Reading recommendations for optimal learning
- Environment setup video tutorial
- LLM development lifecycle overview video

---

### **Chapter 2: Working with Text Data**

- Text tokenization techniques
- Byte-pair encoding (BPE) implementation
- Data loading and preprocessing
- Embedding layers vs matrix multiplication equivalence

**Key Components:**

- `dataloader.ipynb` - Data loading strategies and intuition
- `ch02.ipynb` - Main chapter code with tokenization examples
- Bonus implementations of BPE encoders from scratch
- Exercise solutions with detailed explanations

**Skills Learned:**

- Converting raw text to token sequences
- Creating efficient data loaders for training
- Understanding embedding mechanics

---

### **Chapter 3: Attention Mechanisms**

- Single-head attention mechanisms
- Multi-head attention implementation
- Attention visualization and interpretation
- Masking and scaling in attention

**Key Components:**

- `multihead-attention.ipynb` - Multi-head attention deep dive
- `ch03.ipynb` - Core attention implementation
- Sample text data for experimentation

**Skills Learned:**

- How attention layers compute relationships
- Parallel attention head computation
- Sequence masking for autoregressive models

---

### **Chapter 4: Building GPT Models**

- GPT architecture components
- Model stacking and scaling
- Forward pass implementation
- Loss computation

**Key Components:**

- `gpt.py` - Complete GPT model implementation
- `ch04.ipynb` - Interactive walkthrough
- `tests.py` - Model verification tests
- `previous_chapters.py` - Integrated components from prior chapters

**Skills Learned:**

- Assembling transformer blocks
- Positional encoding
- Token and position embedding combination

---

### **Chapter 5: Pretraining GPT Models**

- Large-scale model training
- Text dataset preparation and downloading
- Training loops and optimization
- Loss tracking and validation
- Generation from pretrained models

**Key Components:**

- `gpt_download.py` - Download pretraining datasets
- `gpt_train.py` - Training pipeline implementation
- `gpt_generate.py` - Inference and text generation
- `ch05.ipynb` - Training walkthrough and examples
- `exercise-solutions.ipynb` - Training exercises with solutions

**Skills Learned:**

- Setting up efficient training loops
- Batch processing and gradient accumulation
- Model checkpointing and resumption
- Generation strategies (greedy, top-k, top-p sampling)

---

### **Chapter 6: Fine-Tuning GPT Models**

- Supervised fine-tuning techniques
- Classification fine-tuning
- Instruction following capabilities
- Transfer learning best practices

**Key Components:**

- `gpt_class_finetune.py` - Classification task fine-tuning
- `gpt_download.py` - Download fine-tuning datasets
- `load-finetuned-model.ipynb` - Loading and using fine-tuned models
- `ch06.ipynb` - Fine-tuning methodology
- `exercise-solutions.ipynb` - Fine-tuning exercises

**Skills Learned:**

- Parameter-efficient fine-tuning
- Task-specific adaptation
- Avoiding catastrophic forgetting
- Evaluation metrics for fine-tuned models

---

### **Chapter 7: Instruction Fine-Tuning & Model Evaluation**

- Instruction-following fine-tuning
- Response generation from instructions
- Model evaluation techniques
- Integration with external tools (Ollama)

**Key Components:**

- `gpt_instruction_finetuning.py` - Instruction dataset preparation and fine-tuning
- `gpt_download.py` - Model and dataset management
- `ollama_evaluate.py` - Model evaluation using Ollama
- `exercise_experiments.py` - Experimental fine-tuning variations
- `load-finetuned-model.ipynb` - Loading instruction-tuned models
- `ch07.ipynb` - Instruction tuning walkthrough
- Training data files:
  - `instruction-data.json` - Instructions without responses
  - `instruction-data-with-response.json` - Complete instruction-response pairs

**Skills Learned:**

- Formatting instruction-response datasets
- Training models to follow arbitrary instructions
- Evaluation frameworks and metrics
- Deployment and serving considerations

---

## 🚀 Getting Started

### **Prerequisites**

- **Python:** 3.9 or higher
- **CUDA:** 11.8+ (for GPU acceleration, recommended for chapters 5+)
- **GPU:** 8GB+ VRAM recommended (can work on CPU for chapters 1-4)
- **Disk Space:** ~10GB (for datasets and model checkpoints)

### **Installation**

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/scratch_llm.git
   cd scratch_llm
   ```

2. **Create Virtual Environment**

   ```powershell
   # Windows
   python -m venv .venv
   .venv\Scripts\Activate.ps1

   # macOS/Linux
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install Dependencies**

   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Installation**
   ```python
   python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

---

## 📁 Project Structure

```
scratch_llm/
├── README.md                          # This file
├── requirements.txt                   # Project dependencies
├── .gitignore                         # Git ignore rules
├── .venv/                             # Virtual environment (not committed)
│
├── ch01/                              # Chapter 1: Understanding LLMs
│   ├── README.md
│   └── reading-recommendations.md
│
├── ch02/                              # Chapter 2: Working with Text Data
│   ├── README.md
│   └── 01_main-chapter-code/
│       ├── ch02.ipynb                 # Main notebook
│       ├── dataloader.ipynb           # Data loading deep dive
│       ├── exercise-solutions.ipynb   # Exercise solutions
│       ├── the-verdict.txt            # Sample text data
│       └── README.md
│
├── ch03/                              # Chapter 3: Attention Mechanisms
│   ├── README.md
│   └── 01_main-chapter-code/
│       ├── ch03.ipynb                 # Main notebook
│       ├── multihead-attention.ipynb  # Attention focus
│       ├── exercise-solutions.ipynb   # Exercise solutions
│       ├── small-text-sample.txt      # Sample data
│       └── README.md
│
├── ch04/                              # Chapter 4: Building GPT Models
│   ├── README.md
│   └── 01_main-chapter-code/
│       ├── ch04.ipynb                 # Main notebook
│       ├── gpt.py                     # GPT implementation
│       ├── previous_chapters.py       # Integrated prior components
│       ├── tests.py                   # Unit tests
│       ├── exercise-solutions.ipynb   # Exercise solutions
│       └── README.md
│
├── ch05/                              # Chapter 5: Pretraining GPT
│   ├── README.md
│   └── 01_main-chapter-code/
│       ├── ch05.ipynb                 # Main notebook
│       ├── gpt_download.py            # Dataset downloading
│       ├── gpt_train.py               # Training pipeline
│       ├── gpt_generate.py            # Generation script
│       ├── previous_chapters.py       # Integrated components
│       ├── tests.py                   # Training tests
│       ├── exercise-solutions.ipynb   # Exercise solutions
│       └── README.md
│
├── ch06/                              # Chapter 6: Fine-Tuning GPT
│   ├── README.md
│   └── 01_main-chapter-code/
│       ├── ch06.ipynb                 # Main notebook
│       ├── gpt_class_finetune.py      # Classification fine-tuning
│       ├── gpt_download.py            # Dataset management
│       ├── load-finetuned-model.ipynb # Loading fine-tuned models
│       ├── previous_chapters.py       # Integrated components
│       ├── tests.py                   # Fine-tuning tests
│       ├── exercise-solutions.ipynb   # Exercise solutions
│       └── README.md
│
└── ch07/                              # Chapter 7: Instruction Fine-Tuning
    ├── README.md
    └── 01_main-chapter-code/
        ├── ch07.ipynb                 # Main notebook
        ├── gpt_instruction_finetuning.py # Instruction tuning
        ├── gpt_download.py            # Model/dataset management
        ├── ollama_evaluate.py         # Model evaluation
        ├── exercise_experiments.py    # Experimental exercises
        ├── load-finetuned-model.ipynb # Loading instruction models
        ├── instruction-data.json      # Instruction data (no responses)
        ├── instruction-data-with-response.json  # Complete instructions
        ├── previous_chapters.py       # Integrated components
        ├── tests.py                   # Evaluation tests
        ├── exercise-solutions.ipynb   # Exercise solutions
        └── README.md
```

---

## 💻 How to Use This Repository

### **For Learning (Recommended Approach)**

1. **Start with Chapter 1** - Read the conceptual overview and watch the setup videos

   ```bash
   cd ch01
   # Read README.md and viewing recommendations
   ```

2. **Progress Chapter by Chapter**
   - Read the chapter README
   - Follow the main Jupyter notebook
   - Run code examples step-by-step
   - Complete exercises and compare with solutions

3. **Chapters 1-3: Foundational Concepts**
   - Work through at your own pace
   - Can run entirely on CPU
   - Focus on understanding concepts deeply

4. **Chapters 4-5: Model Training**
   - GPU recommended but not required
   - May take hours to train models
   - Start with smaller datasets/models for experimentation

5. **Chapters 6-7: Fine-Tuning & Evaluation**
   - Build on trained models from Chapter 5
   - Implement production-ready techniques
   - Learn evaluation strategies

### **Running Jupyter Notebooks**

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook

# Navigate to specific chapter folder and open .ipynb files
```

### **Running Python Scripts**

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Example: Download datasets for Chapter 5
cd ch05/01_main-chapter-code
python gpt_download.py

# Example: Train GPT model
python gpt_train.py

# Example: Generate text with trained model
python gpt_generate.py
```

### **Running Tests**

```powershell
# Activate virtual environment
.venv\Scripts\Activate.ps1

# Run tests for specific chapter
cd ch04/01_main-chapter-code
python -m pytest tests.py -v

# Or run directly
python tests.py
```

---

## 📊 Key Learning Outcomes

By completing this project, you will understand and implement:

✅ **Tokenization & Embeddings** - Text representation and processing
✅ **Attention Mechanisms** - Self-attention and multi-head attention
✅ **Transformer Architecture** - Building blocks of modern LLMs
✅ **Model Training** - Large-scale pretraining pipelines
✅ **Fine-Tuning** - Adapting pretrained models to new tasks
✅ **Instruction Following** - Creating models that follow natural language instructions
✅ **Evaluation** - Assessing model performance
✅ **Best Practices** - Production-ready techniques and considerations

---

## 🛠 Technologies & Dependencies

- **PyTorch** - Deep learning framework
- **Transformers** (Hugging Face) - Tokenizers and utilities
- **NumPy** - Numerical computing
- **Jupyter** - Interactive notebooks
- **Tiktoken** - OpenAI tokenizer
- **Ollama** - Model deployment and evaluation (Chapter 7)

See `requirements.txt` for full dependency list with versions.

---

## ⚙️ Configuration & Tips

### **GPU/CUDA Setup**

For NVIDIA GPUs:

```bash
# Verify CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Check GPU memory
python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0)}'); print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB')"
```

### **Memory Optimization**

For limited GPU memory:

- Reduce `batch_size` in training scripts
- Use `gradient_accumulation_steps` for effective larger batches
- Enable mixed precision training (fp16)
- Use smaller model configurations for experimentation

### **Dataset Management**

Large datasets are downloaded on-demand:

- Chapter 5: OpenWebText or similar pretraining corpus (~500MB-2GB)
- Chapter 6: Classification datasets
- Chapter 7: Instruction-response datasets included

---

## 🐛 Troubleshooting

### **CUDA/GPU Issues**

- Check CUDA compatibility: `nvidia-smi`
- Verify PyTorch CUDA version matches driver
- Reinstall PyTorch if needed: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### **Out of Memory (OOM)**

- Reduce batch size
- Use gradient checkpointing
- Process data in smaller chunks
- Switch to CPU (slower but works)

### **Jupyter Not Starting**

```bash
pip install --upgrade jupyter jupyterlab
jupyter lab --generate-config
jupyter lab
```

### **Notebook Kernel Issues**

```bash
python -m ipykernel install --user --name scratch_llm
# Then select this kernel in Jupyter
```

---

## 📚 Additional Resources

- **Official Chapter Videos** - Links in each chapter's README
- **Reading Recommendations** - [See ch01/reading-recommendations.md](ch01/reading-recommendations.md)
- **PyTorch Documentation** - https://pytorch.org/docs/
- **Hugging Face Documentation** - https://huggingface.co/docs/
- **Transformer Papers** - "Attention Is All You Need", "Language Models are Unsupervised Multitask Learners"

---

## 📝 Exercise Format

Each chapter includes:

- **Main Notebook** - Core concepts and implementations
- **Code-Along Sections** - Step-by-step implementations
- **Exercises** - Hands-on problems to test understanding
- **Solution Notebooks** - Complete solutions with explanations
- **Bonus Materials** - Additional implementations and deep dives

---

## 🤝 Contributing

This is an educational repository. To contribute:

1. Report issues via GitHub Issues
2. Suggest improvements or corrections
3. Share alternative implementations
4. Help improve documentation

---

## 📄 License

This project is provided for educational purposes. Please check the LICENSE file for specific terms.

---

## ✨ Project Highlights

- **7 Comprehensive Chapters** covering LLM fundamentals to advanced techniques
- **Interactive Jupyter Notebooks** with explanations and visualizations
- **Runnable Python Scripts** for training, fine-tuning, and evaluation
- **Test Suites** ensuring code correctness
- **Real-World Datasets** used in production LLM training
- **Best Practices** throughout for scalability and efficiency
- **Progression Path** from theory to practice

---

## 🎯 Quick Start Examples

### **Example 1: Load and Understand GPT Model (Chapter 4)**

```python
import torch
from ch04_01_main_chapter_code.gpt import GPT

# Create model
config = GPT.from_pretrained("gpt2").config
model = GPT(config)
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### **Example 2: Generate Text (Chapter 5)**

```python
from ch05_01_main_chapter_code.gpt_generate import generate

# Generate from trained model
text = generate(model, "Once upon a time", max_tokens=100)
print(text)
```

### **Example 3: Fine-Tune for Classification (Chapter 6)**

```python
from ch06_01_main_chapter_code.gpt_class_finetune import fine_tune

# Fine-tune pretrained model
fine_tune(model, train_dataset, num_epochs=3)
```

---

## 📞 Support & Questions

- Check chapter-specific READMEs for detailed guidance
- Review exercise solutions for similar problems
- Open GitHub Issues for bugs or problems
- Refer to official documentation links provided

---

## 🎓 Learning Path Recommendation

**Week 1-2:** Chapters 1-3 (Concepts & Fundamentals)

- Understanding, tokenization, attention
- Time: ~10-20 hours

**Week 3-4:** Chapter 4 (GPT Architecture)

- Model building and assembly
- Time: ~8-12 hours

**Week 5-8:** Chapter 5 (Pretraining)

- Large-scale training
- Time: ~20-40 hours (depending on data/compute)

**Week 9-10:** Chapters 6-7 (Fine-tuning & Evaluation)

- Practical applications
- Time: ~16-24 hours

**Total:** ~60-100 hours of active learning

---

**Happy Learning! 🚀**

For questions or feedback, please open an issue on GitHub or refer to the chapter-specific resources.
