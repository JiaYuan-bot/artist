# ARTIST: An Agentic Data System With A Learned Optimizer

This repository contains the official implementation and prompts for our VLDB 2026 submission:

> **ARTIST: An Agentic Data System With A Learned Optimizer**  
> Jia Yuan, Chuan Lei, Tim Kraska, Samuel Madden, Lei Cao  
> VLDB 2026


## Repository Structure
```
├── advanced-QA/       # Prompts and code for advanced QA tasks
├── text-to-SQL/       # Prompts and code for text-to-SQL generation
├── text-to-vis/       # Prompts and code for text-to-visualization
└── README.md
```

## Requirements
We recommend using Conda to manage the project environment. For example:
```bash
conda create -n text-to-SQL-env python=3.12.5
conda activate text-to-SQL-env

# install dependencies:
pip install -r requirements.txt
```
You can reuse the same text-to-SQL-env environment for all three tasks, or create separate environments for each task if you prefer to keep dependencies isolated.

## Vertex AI (Google Cloud) Authentication
This project calls LLM APIs through Google Cloud Vertex AI. For local development, set up the Google Cloud CLI and create Application Default Credentials (ADC) so your code can authenticate automatically.
```bash
# 1) Install Google Cloud CLI (Linux)
# https://docs.cloud.google.com/sdk/docs/install-sdk#linux

# 2) Initialize gcloud (select account + default project)
gcloud init

# 3) Create ADC credentials for local usage (opens browser login)
gcloud auth application-default login
```

## Usage
### text-to-SQL
1. You should setup BIRD benchmark first. Please refer to CHESS repo. https://github.com/ShayanTalaei/CHESS

2. run chess_module_service first, it's a RPC service that serves module logics.
    ```bash
    cd test-to-SQL/chess_module_service
    ./run_servers.sh 
    ```
3. run chess_learner, it trains the DQN agent(routing policy)
    ```bash
    cd test-to-SQL/chess_learner
    python main.py
    ```
4. run chess_actor, it run workflow by calling chess_module_service to collect training data for chess_learner
    ```bash
    cd test-to-SQL/chess_actor
    python main.py
    ```

### text-to-Vis
1. You should setup nvAgent benchmark first. Please refer to nvAgent repo. https://github.com/geliang0114/nvAgent

2. run nvAgent_module_service first, it's a RPC service that serves module logics.
    ```bash
    cd test-to-vis/nvAgent_module_service
    ./run_servers.sh 
    ```
3. run nvAgent_learner, it trains the DQN agent(routing policy)
    ```bash
    cd test-to-vis/nvAgent_learner
    python main.py
    ```
4. run nvAgent_actor, it run workflow by calling nvAgent_module_service to collect training data for nvAgent_learner
    ```bash
    cd test-to-vis/nvAgent_actor
    python main.py
    ```


### advanced-QA
1. Please set up the MultiHop-RAG benchmark before running the Advanced-QA task. We already include the code to build and run our retrieval service under:
    ```bash
    advanced-QA/MultiHop-RAG/
    ```
    For dataset details, refer to the upstream MultiHop-RAG repository:
    https://github.com/yixuantt/MultiHop-RAG

2. run qa_module_service first, it's a RPC service that serves module logics.
    ```bash
    cd advanced-QA/qa_module_service
    ./run_servers.sh 
    ```
3. run qa_learner, it trains the DQN agent(routing policy)
    ```bash
    cd advanced-QA/qa_learner
    python main.py
    ```
4. run nvAgent_actor, it run workflow by calling qa_module_service to collect training data for qa_learner
    ```bash
    cd advanced-QA/qa_actor
    python main.py
    ```

## LLM as Planner Baseline
You can find prompts used for LLM as planner baseline at:
```
test-to-SQL/chess_actor/rl_distributed/llm_planner
test-to-vis/nvAgent_actor/rl_distributed/llm_planner
advanced-QA/qa_actor/rl_distributed/llm_planner
```

