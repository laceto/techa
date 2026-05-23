# ML & AI Communities — Research, Models, Experimentation

Platforms where the community itself is the product: shared models, datasets, experiments, papers, and benchmarks.

---

## Hugging Face

**URL:** https://huggingface.co

The central hub of the open-source AI world. Every major model is here.

### Sign Up
- Free account: access models, datasets, Spaces
- Pro ($9/month): private models, faster inference, ZeroGPU priority
- Enterprise: on-premise Hub, SSO, audit

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Models Hub** | 800 000+ model checkpoints (Llama, Mistral, Stable Diffusion, BERT…) |
| **Datasets Hub** | 200 000+ datasets, one-line download |
| **Transformers** | `pip install transformers` — run any model in 5 lines |
| **Inference API** | Call any model via REST without deploying anything |
| **Spaces** | Deploy Gradio or Streamlit apps for free (GPU available) |
| **Inference Endpoints** | Dedicated, private model endpoint (paid) |
| **AutoTrain** | No-code fine-tuning on your dataset |
| **Diffusers** | Stable Diffusion, SDXL, video diffusion pipelines |
| **PEFT / TRL** | Efficient fine-tuning (LoRA, QLoRA) and RLHF |

```python
from transformers import pipeline
pipe = pipeline("text-generation", model="meta-llama/Llama-3.2-1B-Instruct")
print(pipe("The future of AI is")[0]["generated_text"])
```

### Community
- **Papers** tab: link any arXiv paper to its code
- **Discussions** on every model and dataset
- **Organizations**: join or create team spaces
- **Leaderboards**: Open LLM Leaderboard, MTEB, etc.
- Discord: `hf.co/join/discord`

### Ship
- **Spaces** — public Gradio/Streamlit demo in minutes, free CPU, paid GPU
- **Inference Endpoints** — production API, auto-scaling
- **Docker Spaces** — deploy any containerized app
- Share model card → community can use your model immediately

### Full Journey Example
```
Sign up → find Mistral-7B → run in Inference API playground
→ Fine-tune with AutoTrain on your data
→ Deploy as Space (Gradio UI) or Endpoint (REST API)
→ Share model card — community uses your model
```

---

## Kaggle

**URL:** https://kaggle.com  (owned by Google)

### Sign Up
- Free: full access to notebooks (GPU/TPU), datasets, competitions
- Google account sign-in supported
- Kaggle progression tiers: Novice → Contributor → Expert → Master → Grandmaster

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Notebooks** | Jupyter notebooks with free GPU (P100 / T4) and TPU |
| **Datasets** | 300 000+ public datasets, instant mount into notebook |
| **Models Hub** | Host and share model weights (integrated with TF Hub, HF) |
| **Competitions** | Cash-prize ML challenges (image, NLP, tabular, RL) |
| **Courses** | Free ML/DL/AI courses with coding exercises |
| **Kaggle AI Report** | Community-generated AI insights |

```python
# Typical competition kernel start
import pandas as pd
train = pd.read_csv("/kaggle/input/competition-name/train.csv")
# GPU available, all major ML libs pre-installed
```

### Community
- Competition forums: team up, share notebooks, discuss approaches
- Discussion boards per dataset
- Notebook voting and awards
- Kaggle Discord

### Ship
- **Competition submission** — submit predictions, get on the leaderboard
- **Published notebooks** — your work is public and citable
- **Datasets** — publish data for others to use
- **Models** — share trained weights in the Models Hub

---

## Papers with Code

**URL:** https://paperswithcode.com

### Sign Up
- Browse without account
- GitHub login to submit papers, claim results, join discussions

### Code & AI
| Feature | What it does |
|---------|-------------|
| **State-of-the-art** | Live SOTA tables for every task and benchmark |
| **Paper + Code links** | Every paper linked to its GitHub repo |
| **Methods library** | Searchable catalogue of ML techniques |
| **Datasets** | Curated benchmark datasets with leaderboards |
| **Trending** | What the community is reading and reproducing now |

### Community
- Submit papers: link your arXiv paper to your GitHub
- Claim results: update SOTA tables with your numbers
- Community corrections and additions

### Ship
- Primarily a research publication channel
- Publishing here makes your work discoverable by practitioners
- Links drive GitHub stars, citations, and adoption

---

## Weights & Biases (W&B)

**URL:** https://wandb.ai

### Sign Up
- Free: unlimited personal projects, 100 GB storage
- Team plans: shared dashboards, access control
- Academic: free Team tier

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Experiment Tracking** | Log metrics, hyperparameters, gradients, media per run |
| **Sweeps** | Automated hyperparameter search (grid, random, Bayes) |
| **Artifacts** | Version datasets, models, and evaluation results |
| **Model Registry** | Promote models from experiments → staging → production |
| **Weave** | LLM observability: trace calls, eval datasets, score outputs |
| **Reports** | Shareable analysis docs with live charts |

```python
import wandb
wandb.init(project="my-llm", config={"lr": 1e-4, "model": "llama-3"})
for step, loss in enumerate(training_losses):
    wandb.log({"loss": loss, "step": step})
wandb.finish()
```

### Community
- **Fully Connected** blog: ML practitioner content
- W&B Reports Gallery: browse public experiments
- Discord: active ML community
- Courses: free W&B ML courses

### Ship
- Model Registry → trigger deployment pipelines
- Weave → monitor LLM apps in production
- Reports → shareable reproducible research

---

## LangChain / LangSmith

**URL:** https://langchain.com | https://smith.langchain.com

### Sign Up
- LangChain: open-source library, no account needed
- LangSmith: sign up free (developer tier generous)
- LangChain Plus: paid for higher trace volume

### Code & AI
| Feature | What it does |
|---------|-------------|
| **LangChain** | Framework for LLM apps: chains, agents, RAG, tools |
| **LangGraph** | Stateful agent orchestration with graph-based flow |
| **LangSmith** | Trace, debug, evaluate every LLM call |
| **LangServe** | Deploy any LangChain chain as a FastAPI endpoint |
| **Hub** | Share and reuse prompts (`langchain hub pull`) |
| **Integrations** | 600+ integrations: every LLM, vector DB, tool |

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

chain = ChatPromptTemplate.from_template("Tell me about {topic}") | ChatAnthropic(model="claude-sonnet-4-6")
print(chain.invoke({"topic": "vector databases"}).content)
```

### Community
- GitHub: `github.com/langchain-ai/langchain` (100k+ stars)
- Discord: large active server
- LangChain Blog: tutorials and releases
- Prompt Hub: community-shared prompts

### Ship
- **LangServe** → REST API from your chain in one decorator
- **LangGraph Platform** → deploy stateful agents with persistence
- Self-host or deploy to any cloud
