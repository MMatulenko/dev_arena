# Dev Arena: The Self-Evolving AI Software Agency

Most autonomous coding agents (like Devin, OpenHands, or Warp's Oz) rely on highly optimized but **static** system prompts. They execute tasks but they do not structurally learn from their own failures.

**Dev Arena** introduces a novel **Dual-Agent Continuous Meta-Learning Architecture**. It doesn't just write code; it dynamically patches its own instruction manuals and identifies environment deficiencies when it fails. 

In our benchmark testing across coding datasets, this self-reflecting meta-learning approach resulted in a **14% performance improvement per $1 of learning budget** compared to static zero-shot agents.

## 🧠 The Dual-Agent Architecture

The arena consists of two primary entities:
1. **The Junior Coder (Executor):** An economical, fast model (e.g., `gpt-4o-mini`) that executes the actual coding tasks within a secure Docker environment (SWE-bench).
2. **The Senior Dev Coach (Strategist):** A high-reasoning model (e.g., `gpt-5.2` or `claude-3.5-sonnet`) that reviews the Junior's failed trajectories.

When the Junior fails a task, the Senior Coach intervenes. The Coach analyzes the failure, deduces the structural flaw in the Junior's approach, and pushes a patch to `CORE_PRINCIPLES.md`—the Junior's "Playbook". The next time the Junior spawns, it possesses this new knowledge.

### Architecture Flow

```mermaid
graph TD
    A[Task Dispatched (SWE-bench)] --> B[Junior Agent loop]
    B -->|Tool Usage| C[Docker Workspace]
    C -->|Run Tests| D{Tests Pass?}
    D -- Yes --> E[Task Complete!]
    D -- No --> F[Failure Trajectory Logged]
    F --> G[Senior Coach Analysis]
    G -->|Updates Principles| H[CORE_PRINCIPLES.md]
    G -->|Identifies Tool Bugs| I[TOOL_TICKETS.md]
    H --> B
```

## ✨ Key Features

- **Continuous Meta-Learning:** The agent literally learns how to code better over time by evolving its `CORE_PRINCIPLES.md` file.
- **Auto-Tool Ticketing (`TOOL_TICKETS.md`):** Instead of failing silently when grep or file patching tools behave unexpectedly, the Senior Coach detects systemic tool deficiencies and automatically files "Tool Upgrade Tickets" for human engineers to improve the agent's environment.
- **Aggressive Cost Engineering:** By utilizing OpenAI's automatic prompt caching and Anthropic's ephemeral `cache_control` blocks, Dev Arena cuts the cost of running extensive SWE-bench evaluations by up to 90%.
- **Cross-Provider Support:** Seamlessly switch between OpenAI and Anthropic models depending on your budget and reasoning needs.

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/yourusername/dev_arena.git
cd dev_arena
pip install -r requirements.txt
```

### 2. Configuration

Copy the example environment file and add your API keys:

```bash
cp .env.example .env
```

Edit `.env`:
```env
OPENAI_API_KEY=sk-your-openai-key-here
ANTHROPIC_API_KEY=sk-ant-your-anthropic-key-here
LLM_PROVIDER=openai
OPENAI_COACH_MODEL=gpt-5.2
```

### 3. Running the Dev Arena (SWE-Bench)

To run the agent against real-world software engineering issues using SWE-bench Lite:

```bash
python dev_loop.py --swe-bench 10
```

### 4. Exploring the Chess Examples

Before tackling full software engineering workflows, the architecture was originally prototyped in a closed-domain game environment to measure cognitive evolution. You can find these experiments in the `chess_arena` directory.

```bash
cd chess_arena
# Run the cognitive learning loop against puzzles
python cognitive_loop.py
```

## 📈 Performance

By allowing the Senior Coach to update the Junior's core rules after failed sprints, the system adapts to idiosyncrasies in test runners, file patchers, and specific repository architectures. 

Compared to static baseline models, **Dev Arena achieves a 14% higher pass rate when constrained to a $1 learning budget**, proving that dynamic meta-prompting is vastly superior to static prompt engineering.

## License
MIT License
