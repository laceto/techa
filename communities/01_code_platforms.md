# Code Platforms with Embedded AI

Platforms where the IDE, the AI assistant, and the deployment pipeline live in one place — you sign up and the loop is closed.

---

## GitHub

**URL:** https://github.com

### Sign Up
- Free personal account (unlimited public + private repos)
- GitHub Pro / Team / Enterprise for advanced features
- Student Developer Pack: free Pro + extras for students

### Code & AI
| Feature | What it does |
|---------|-------------|
| **GitHub Copilot** | Inline code completion, chat in editor, code review |
| **Copilot Workspace** | Task-to-PR: describe a task, Copilot writes plan + code |
| **GitHub Codespaces** | Full VS Code in the browser, runs your repo in a container |
| **GitHub Actions** | CI/CD pipelines — test, build, deploy on every push |
| **GitHub Models** | Playground to test GPT-4o, Llama, Mistral directly from GitHub |

### Community
- Issues and Discussions on every repo
- Pull Requests with inline review
- Stars, forks, sponsors
- GitHub Marketplace: 10 000+ Actions and Apps
- GitHub Education: campus community

### Ship
- **GitHub Pages** — static site from any repo, free
- **GitHub Actions** → deploy to any cloud (AWS, GCP, Azure, Vercel, etc.)
- **GitHub Releases** — versioned binaries and packages
- **GitHub Packages** — private npm, Docker, Maven registry

### Full Journey Example
```
Sign up → Create repo → Open Codespace → Copilot writes code
→ Push → Actions run tests → Actions deploy to Pages or cloud
→ Tag a release → done
```

---

## Replit

**URL:** https://replit.com

### Sign Up
- Free tier: unlimited public repls, limited compute
- Core ($20/month): private repls, more CPU/RAM, deployments
- Teams: shared workspace for orgs

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Replit Agent** | Describe an app in plain English → Agent writes, runs, debugs it |
| **Ghostwriter** | Copilot-style inline code completion |
| **50+ languages** | Python, Node.js, Go, Rust, C++, Java… |
| **Multiplayer** | Real-time collaborative coding (like Google Docs for code) |
| **Nix environment** | Full Linux shell, install any package |

### Community
- Replit Community: share repls, get feedback
- Templates marketplace: start from 1 000+ starters
- Replit Bounties: paid tasks posted by companies
- Discord server: `discord.gg/replit`

### Ship
- **Replit Deployments** — one click, always-on server (Core tier)
- **Custom domains** — connect your own domain
- **Static deployments** — free for front-end only
- Auto-scales, no infra config required

### Full Journey Example
```
Sign up → "Build me a FastAPI todo app with SQLite"
→ Agent scaffolds project → run in browser → Deploy
→ Share the URL — done
```

---

## Bolt.new / StackBlitz

**URL:** https://bolt.new

### Sign Up
- Free tier with daily token budget
- Paid plans for heavier usage
- Powered by StackBlitz (WebContainers — Node.js in the browser)

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Bolt AI** | Prompt → full-stack app (React, Next.js, Vite, Astro…) |
| **WebContainers** | Node.js runs entirely in the browser, zero install |
| **File editor** | Full file tree, edit any file AI generated |
| **Terminal** | Real shell inside the browser tab |
| **npm support** | Install any npm package inside the browser |

### Community
- StackBlitz community: share project links
- Twitter / X: `#boltai` hashtag
- Discord: StackBlitz server

### Ship
- **Download as ZIP** → deploy anywhere
- **Netlify / Vercel integration** — connect and deploy in one click
- **StackBlitz.com** — sharable live URL for any project

### Full Journey Example
```
Go to bolt.new (no account needed to start)
→ "Create a SaaS landing page with Tailwind and dark mode"
→ Edit generated code → Deploy to Netlify
→ done
```

---

## Cursor

**URL:** https://cursor.sh

### Sign Up
- Free tier: 2 000 Cursor Tab completions / month, limited AI chat
- Pro ($20/month): unlimited completions, GPT-4o, Claude Sonnet
- Business: team management, privacy mode

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Cursor Tab** | Predictive multi-line completions (beyond Copilot) |
| **Cursor Chat** | Chat with your codebase — ask questions about any file |
| **Composer** | Multi-file AI edits from a single prompt |
| **`@` context** | Pin files, docs, web URLs into the AI context |
| **Model choice** | GPT-4o, Claude 3.5 Sonnet, Gemini, local models |
| **Rules for AI** | `.cursorrules` — define coding standards the AI follows |

### Community
- Cursor Forum: `forum.cursor.sh`
- Discord: active server for tips and prompts
- GitHub: community `.cursorrules` collections

### Ship
- Cursor is an editor (fork of VS Code) — use any deployment pipeline
- Works with GitHub Actions, Vercel CLI, Railway, etc.
- Pairs with any language / framework

### Full Journey Example
```
Download Cursor → open project → Composer:
"Refactor this Express app to use Hono, add Zod validation"
→ Review diffs → accept → git push → CI deploys
```

---

## Windsurf (by Codeium)

**URL:** https://codeium.com/windsurf

### Sign Up
- Free tier available
- Pro for heavier model usage
- Enterprise: SSO, audit logs, self-hosted models

### Code & AI
| Feature | What it does |
|---------|-------------|
| **Cascade** | Agentic mode — AI acts on your codebase autonomously |
| **Flows** | AI + human collaborate in the same edit stream |
| **Supercomplete** | Context-aware multi-line completions |
| **Command** | Natural language → terminal commands |

### Community
- Codeium Discord
- Windsurf Extensions Marketplace

### Ship
- VS Code-compatible, use any deployment pipeline
- Integrates with GitHub, GitLab, Bitbucket
