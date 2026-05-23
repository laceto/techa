# Deployment & Shipping Platforms

Where your code + AI becomes a live product accessible to users. Covers the final step of the journey.

---

## Vercel + v0.dev

**URL:** https://vercel.com | https://v0.dev

### Sign Up
- Vercel: free Hobby plan (personal projects, generous limits)
- Pro ($20/month): team, more bandwidth, preview environments
- v0.dev: integrated with Vercel account, message-based credits

### Code & AI
| Feature | What it does |
|---------|-------------|
| **v0.dev** | Prompt → React/Tailwind UI component, copy or deploy directly |
| **Vercel AI SDK** | Streaming AI responses, tool use, RAG in Next.js apps |
| **Next.js** | Vercel's own framework — App Router, Server Actions |
| **Edge Functions** | AI inference at the edge (low latency globally) |
| **Fluid Compute** | Long-running serverless for AI inference |
| **Analytics** | Real user metrics built in |
| **Preview Deployments** | Every PR gets its own live URL |

```bash
# Deploy a Next.js AI app in 3 commands
npx create-next-app@latest my-ai-app
cd my-ai-app
vercel deploy
```

### Community
- Vercel community forum
- GitHub: Next.js repo (250k+ stars)
- v0.dev: share your generated UIs
- Templates gallery: 100+ production starters

### Ship
- Git push → auto-deploy (GitHub / GitLab / Bitbucket)
- Custom domains with automatic HTTPS
- Global CDN with 40+ edge locations
- From v0 UI prompt to live URL in under 2 minutes

### Full Journey Example
```
v0.dev: "Build a chat UI with sidebar and dark mode"
→ Copy generated code into Next.js project
→ Add Vercel AI SDK + OpenAI key
→ git push → Vercel auto-deploys
→ Share custom domain — done
```

---

## Streamlit Community Cloud

**URL:** https://streamlit.io/cloud

### Sign Up
- Free: unlimited public apps, connect GitHub
- Paid workspace: private apps, more resources
- Part of Snowflake ecosystem

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Streamlit** | Python → interactive web app, no HTML/CSS/JS |
| **`st.chat_message`** | Built-in chat UI for LLM apps |
| **`st.write_stream`** | Stream LLM tokens directly to the UI |
| **Secrets management** | Store API keys securely in Community Cloud |
| **st-aggrid, Plotly** | Rich interactive data tables and charts |

```python
import streamlit as st
from anthropic import Anthropic

client = Anthropic(api_key=st.secrets["ANTHROPIC_API_KEY"])
if prompt := st.chat_input("Ask anything"):
    with st.chat_message("assistant"):
        with client.messages.stream(model="claude-sonnet-4-6", max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]) as stream:
            st.write_stream(stream.text_stream)
```

### Community
- Streamlit Forum: `discuss.streamlit.io`
- Streamlit Gallery: public apps for inspiration
- Discord: active Python + ML community
- Components library: 150+ community components

### Ship
- Connect GitHub repo → deploy in 2 clicks
- Free HTTPS URL: `your-app.streamlit.app`
- Auto-redeploy on every git push
- Ideal for ML demos, internal tools, data dashboards

---

## Railway

**URL:** https://railway.app

### Sign Up
- Free starter: $5 credit / month
- Pro ($20/month): more compute, priority support
- No credit card required to start

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Any language** | Python, Node, Go, Rust, Java — deploy from repo |
| **Databases** | Postgres, MySQL, Redis, MongoDB — one click |
| **Private networking** | Services talk to each other over private network |
| **Cron jobs** | Scheduled tasks built in |
| **Volume storage** | Persistent disk for model weights, uploads |
| **Nixpacks** | Auto-detect language and build — zero config |

```bash
# Deploy a Python API
git add . && git commit -m "init"
railway login
railway up
# Railway detects Python, installs deps, deploys
```

### Community
- Railway Discord (very active)
- Railway blog: deployment tutorials
- Community templates: FastAPI + Postgres, Next.js, etc.

### Ship
- Full-stack: front-end + back-end + database in one project
- Custom domains, automatic TLS
- Deploy AI apps with GPU instances (via add-on)
- Environment variables → secrets management

---

## Render

**URL:** https://render.com

### Sign Up
- Free tier: web services (sleep after 15 min inactivity), static sites, cron
- Paid: always-on instances from $7/month

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Web Services** | Any Docker or Git-based app |
| **Static Sites** | Free, global CDN |
| **Background Workers** | Long-running ML inference jobs |
| **Cron Jobs** | Scheduled model runs |
| **Postgres** | Managed database |
| **Private Services** | Internal microservices not exposed to internet |

### Community
- Render community forum
- GitHub: example repos for common stacks

### Ship
- Git push → auto-deploy pipeline
- Preview environments for PRs
- Good fit: FastAPI AI backend, Celery workers for async inference

---

## Fly.io

**URL:** https://fly.io

### Sign Up
- Free allowance: 3 shared-CPU VMs, 3 GB storage
- Pay-as-you-go for more

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Machines API** | Launch GPU machines on demand for inference |
| **Docker-native** | Deploy any container |
| **Global regions** | 30+ regions — run close to your users |
| **Persistent volumes** | Store model weights across restarts |
| **WireGuard VPN** | Private networking between services |

### Ship
- Best for: containerized AI APIs that need GPU burst
- `fly launch` auto-detects app type
- Scale to zero when idle
