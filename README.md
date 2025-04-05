

## Setup (Ubuntu)

### 1 Update System and Install Python 3.12
```bash
sudo apt update
sudo apt install python3.12-venv
sudo apt install python3.12
```

### 2 Create Virtual Environment
```bash
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3 Install UV (Ultra Fast Python Package Manager)
```bash
pip install uv
```

### 4 Install Project Dependencies
```bash
uv pip install -r pyproject.toml
```

### 5 Update .env file
```bash
cp .env.example .env
```
Write your API keys to .env file
---

### 5 Run The Application
```bash
python main.py
```
---