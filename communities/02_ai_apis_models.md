# AI APIs & Model Platforms

Platforms where the core product is access to AI models — you sign up, get an API key, build on top of the models, and ship via their infrastructure or your own.

---

## OpenAI Platform

**URL:** https://platform.openai.com

### Sign Up
- Personal account at platform.openai.com
- Free $5 credit on new accounts (limited time)
- Pay-as-you-go after free credit; no monthly fee required
- ChatGPT Plus ($20/month) for the consumer product

### Code & AI
| Model / Feature | What it does |
|----------------|--------------|
| **GPT-4o** | Text, vision, audio — the flagship multimodal model |
| **o1 / o3** | Reasoning models for math, science, complex coding |
| **DALL-E 3** | Text-to-image generation |
| **Whisper** | Speech-to-text (open-source weights available) |
| **TTS** | Text-to-speech, 6 voices |
| **Embeddings** | `text-embedding-3-small/large` for vector search |
| **Fine-tuning** | Custom GPT-4o / GPT-4o mini on your data |
| **Assistants API** | Stateful agents with tools (code interpreter, file search) |
| **Batch API** | Async bulk requests at 50% cost |

```python
# Minimal example — 3 lines to call GPT-4o
from openai import OpenAI
client = OpenAI()  # reads OPENAI_API_KEY from env
print(client.chat.completions.create(model="gpt-4o", messages=[{"role":"user","content":"Hello"}]).choices[0].message.content)
```

### Community
- OpenAI Community Forum: `community.openai.com`
- OpenAI Discord (via waitlist)
- OpenAI Cookbook: `github.com/openai/openai-cookbook`
- GPT Store: publish and discover custom GPTs

### Ship
- **GPT Store** — publish a Custom GPT to millions of ChatGPT Plus users
- **Assistants API** — embed a stateful assistant in your app
- **API endpoints** — build any SaaS on top; you own the deployment
- **ChatGPT plugins / Actions** — extend ChatGPT with your own API

---

## Anthropic / Claude

**URL:** https://console.anthropic.com | https://claude.ai

### Sign Up
- claude.ai — consumer chat (free + Pro $20/month)
- console.anthropic.com — API access, pay-as-you-go
- Claude for Work / Enterprise — team features

### Code & AI
| Model | Context | Strength |
|-------|---------|----------|
| **Claude Opus 4.7** | 200K tokens | Deep reasoning, complex tasks |
| **Claude Sonnet 4.6** | 200K tokens | Best speed/quality balance |
| **Claude Haiku 4.5** | 200K tokens | Fastest, cheapest |

| Feature | What it does |
|---------|-------------|
| **Extended Thinking** | Visible chain-of-thought before answering |
| **Tool Use** | Function calling / tool use in multi-turn agents |
| **Computer Use** | Claude controls a virtual desktop (beta) |
| **Files API** | Upload PDFs, docs, images into context |
| **Prompt Caching** | Cache long system prompts → 90% cost reduction |
| **Claude Code (CLI)** | AI pair programmer in your terminal |

```python
from anthropic import Anthropic
client = Anthropic()
msg = client.messages.create(model="claude-sonnet-4-6", max_tokens=1024,
    messages=[{"role": "user", "content": "Write a Python web scraper"}])
print(msg.content[0].text)
```

### Community
- Anthropic Discord
- Claude Prompt Library: `docs.anthropic.com/prompt-library`
- GitHub: `github.com/anthropics`

### Ship
- Build any product via the Messages API
- Claude Code: AI dev tool deployed in your workflow
- Bedrock / GCP Vertex: run Claude on AWS or Google Cloud

---

## Google AI Studio & Vertex AI

**URL:** https://aistudio.google.com | https://cloud.google.com/vertex-ai

### Sign Up
- Google account → AI Studio (free, generous free tier)
- Google Cloud account → Vertex AI (pay-as-you-go, $300 free credit)

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Gemini 2.5 Pro/Flash** | Multimodal: text, image, video, audio, code |
| **Gemini 1M context** | The longest context window available (1 million tokens) |
| **Imagen 3** | State-of-the-art text-to-image |
| **Veo 2** | Text-to-video generation |
| **NotebookLM** | AI-powered research notebook |
| **Google Colab** | Free Jupyter notebooks with GPU/TPU |
| **Vertex AI Workbench** | Managed Jupyter on GCP |
| **AutoML** | No-code model training |
| **Vertex AI Pipelines** | MLOps orchestration |

```python
import google.generativeai as genai
genai.configure(api_key="YOUR_KEY")
model = genai.GenerativeModel("gemini-2.5-pro")
print(model.generate_content("Explain transformers").text)
```

### Community
- Google ML Community: `developers.google.com/community`
- TensorFlow / Keras community
- Google AI Discord
- Kaggle (Google-owned) — see `03_ml_communities.md`

### Ship
- **Vertex AI Endpoints** — serve models at scale
- **Cloud Run** — containerized app deployment
- **Firebase + Gemini** — mobile/web app with AI backend
- **Google Cloud Functions** — serverless AI microservices

---

## Together AI

**URL:** https://together.ai

### Sign Up
- Free account: $5 starting credit
- Pay-as-you-go: per-token pricing
- Enterprise: dedicated capacity

### Code & AI
| Feature | What it does |
|---------|-------------|
| **100+ open models** | Llama 3.x, Mistral, Mixtral, Qwen, DBRX, Gemma… |
| **OpenAI-compatible API** | Drop-in replacement — change base URL only |
| **Fine-tuning** | Fine-tune Llama / Mistral on your data |
| **Embeddings** | M2-BERT and other open embedding models |
| **Vision models** | LLaVA, InternVL |
| **Dedicated instances** | Reserved GPU for production latency |

```python
from openai import OpenAI
client = OpenAI(api_key="TOGETHER_KEY", base_url="https://api.together.xyz/v1")
response = client.chat.completions.create(
    model="meta-llama/Llama-3.3-70B-Instruct-Turbo",
    messages=[{"role": "user", "content": "Hello"}]
)
```

### Community
- Together Discord
- Open-source model leaderboard on their site
- Together Blog: model release posts, benchmarks

### Ship
- REST API → integrate into any app
- Fine-tuned model → dedicated endpoint
- Together Inference: production-grade, SLA-backed

---

## Replicate

**URL:** https://replicate.com

### Sign Up
- Free account: limited free predictions
- Pay-as-you-go: per second of GPU use
- No credit card required to explore models

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Model zoo** | 1 000+ open models (Stable Diffusion, Llama, Whisper, SDXL…) |
| **Python / Node client** | `replicate.run("model", input={})` |
| **cog** | Containerize your own model in one YAML file |
| **Deployments** | Your model as a private always-on endpoint |
| **Streaming** | Stream text tokens in real time |

```python
import replicate
output = replicate.run(
    "meta/meta-llama-3-8b-instruct",
    input={"prompt": "Write a haiku about code"}
)
print("".join(output))
```

### Community
- Replicate Explore: browse and comment on models
- GitHub: open-source cog tool
- Twitter / X: active community

### Ship
- Publish a model → public URL anyone can call
- **Deployments** → private endpoint, auto-scaling
- Webhook support → async production workflows

---

## Groq

**URL:** https://groq.com | https://console.groq.com

### Sign Up
- Free tier: generous rate limits for prototyping
- Pay-as-you-go: among the cheapest per-token rates

### Code & AI
| Feature | What it does |
|---------|-------------|
| **LPU hardware** | 10-100x faster inference than GPU (Groq Language Processing Unit) |
| **Llama 3.x** | Meta's models at blazing speed |
| **Mixtral / Gemma** | Open models, same ultra-low latency |
| **Whisper** | Fastest speech-to-text available |
| **OpenAI-compatible** | Same SDK, swap base URL |

```python
from groq import Groq
client = Groq()
print(client.chat.completions.create(
    model="llama-3.3-70b-versatile",
    messages=[{"role": "user", "content": "Explain RAG in one paragraph"}]
).choices[0].message.content)
```

### Community
- Groq Discord
- GroqCloud docs and changelog

### Ship
- API integration into any product
- Ideal for real-time applications (voice bots, live coding assistants)
- No deployment infra required — just the API
