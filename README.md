## Setup Instructions

### 1. Clone the repository

```bash
git clone https://github.com/maitreyyi/cs121searchengine.git
cd cs121searchengine
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv

Unix: source venv/bin/activate  
Windows: venv\Scripts\activate
Mac: venv activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Download and extract dataset
Download developer.zip and extract into /data

### 5.  Run indexer
```bash
python index.py
```