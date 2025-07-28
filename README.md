# 🚀 Challenge 1A – PDF Processing Solution (Adobe India Hackathon 2025)

This repository contains our submission for **Challenge 1A** of the **Adobe India Hackathon 2025**.
It provides a PDF processing pipeline that extracts structured hierarchical data (headings and titles) from input PDF documents and generates JSON output conforming to the provided schema.

---

## 📁 Folder Structure

```
Challenge_1A/
├── Dockerfile
├── process_pdfs.py
├── requirements.txt
├── README.md
├── input/                        # Mount point for PDFs and input JSON
├── output/                       # Mount point for output JSON
├── models/
│   └── e5-small-v2/              # Offline Hugging Face model files
│       ├── config                
│       ├── model.safetensors    
│       ├── special_tokens_map    
│       ├── tokenizer             
│       ├── tokenizer_config      
│       └── vocab                 

```

---

## 🐳 Build Instructions

The solution is containerized using Docker and must be built using the following command:

```bash
docker build --platform linux/amd64 -t pdf-processor .
```

This will:

* Use the `python:3.13` base image
* Install a CPU-only version of PyTorch
* Install all required packages from `requirements.txt`
* Copy the processing script and model files into the container
* Set the script as the entrypoint for automatic execution

---

## ▶️ Run Instructions

Once built, run the solution using the command:

```bash
docker run --rm \
  -v $(pwd)/input:/app/input:ro \
  -v $(pwd)/output:/app/output \
  --network none \
  pdf-processor
```

Replace:

* `$(pwd)/input` with the path to your folder containing `.pdf` files
* `$(pwd)/output` with the path to an empty folder for `.json` output

👍 All PDFs in `/app/input` will be processed
👍 A corresponding `.json` file will be generated in `/app/output` for each `.pdf`

---

## 📦 Output Format

Each output JSON will follow this structure:

```json
{
  "title": "Document Title",
  "outline": [
    {
      "level": "H1",
      "text": "Introduction",
      "page": 0
    },
    {
      "level": "H2",
      "text": "Overview",
      "page": 1
    }
    // ...
  ]
}
```

This format conforms to the schema provided in
`sample_dataset/schema/output_schema.json`.

---

## 🧠 Solution Details

**Model Used:** `intfloat/e5-small-v2`, embedded locally in `/models/e5-small-v2`
**Libraries Used:**
- transformers
- torch
- PyMuPDF (fitz)
- numpy
- re (regex), collections, pathlib, json


### 📌 Heading Detection Logic

* Combines font size, boldness, spacing, and text patterns
* Falls back to semantic similarity using embedding-based classification when font signals are weak
* Supports hierarchical classification into `"H1"`, `"H2"`, `"H3"` levels
* Excludes unlikely headings:
  * Blocks that start with a number
  * Blocks that are only one character long
  * Blocks that are standalone special characters (e.g., `*`, `-`, `#`, etc.)

### 🏧 Title Extraction

* Uses heuristics based on page position, font size, and keyword patterns

### 🔌 Offline Support

* Model files are copied into the Docker image
* `local_files_only=True` ensures full offline execution

---

## ⚙️ Requirements

The solution requires:

* ❌ No internet access during runtime
* 💻 Only CPU usage (no GPU)
* 📆 Model size < 200MB
* 🧠 Memory usage < 16GB
* ⚡ Execution within 10 seconds for a 50-page PDF
* 📜 All dependencies are open source and listed in `requirements.txt`

---

## ✅ Validation Checklist

* All PDFs in `/app/input` are processed
* Each `.pdf` file results in a `.json` file in `/app/output`
* Output JSON matches the provided schema
* No internet access required (offline execution)
* Compatible with `linux/amd64` architecture
* Runs on 8 CPUs with 16GB RAM or less

---

## 📃 Author(s)

* **Team Name:** Code-Blooded
* **Team Members:** Rohit N, Aayush Menon
* **Submission:** Challenge 1A – PDF Processing
* **Institution:** Vellore Institute of Technology (VIT), Chennai Campus
* **Hackathon:** Adobe India Hackathon 2025
