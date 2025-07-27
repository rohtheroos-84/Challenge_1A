import json
import torch
import torch.nn.functional as F
import re
import fitz 
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
from collections import Counter
from pathlib import Path

device = "cpu"
MODEL_NAME = "/app/models/e5-small-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME,trust_remote_code=True,local_files_only=True)
model = AutoModel.from_pretrained(MODEL_NAME,trust_remote_code=True,local_files_only=True).to(device).eval()

@torch.no_grad()
def embed(texts: list[str]) -> torch.Tensor:
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt").to(device)
    out = model(**inputs)
    return out.last_hidden_state[:, 0, :]

def clean_repeated_text(text: str) -> str:
    words = text.split()
    if not words: return ""
    n = len(words)
    for length in range(n // 2, 0, -1):
        if n % length == 0:
            num_repeats = n // length
            if words == words[:length] * num_repeats:
                return " ".join(words[:length])
    return text

def extract_blocks(doc):
    blocks = []
    for page_num, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue

            span_chunks = []
            chunk = {
                "text": "",
                "font_sizes": [],
                "is_bold": True
            }

            last_font_size = None

            for line in block["lines"]:
                for span in line["spans"]:
                    text = span["text"].strip()
                    if not text:
                        continue

                    font_size = span["size"]
                    is_bold = ((span["flags"] & 2**4) > 0) or ("bold" in span.get("font", "").lower())

                    if chunk["text"] and (abs(font_size - last_font_size) >= 2 or chunk["is_bold"] != is_bold or text[0].isupper()):
                        span_chunks.append(chunk)
                        chunk = {"text": "", "font_sizes": [], "is_bold": True}

                    chunk["text"] += text + " "
                    chunk["font_sizes"].append(font_size)
                    chunk["is_bold"] &= is_bold
                    last_font_size = font_size


            if chunk["text"].strip():
                span_chunks.append(chunk)

            for chunk in span_chunks:
                cleaned_text = clean_repeated_text(chunk["text"].strip())

                if len(cleaned_text) > 30:
                    tokens = cleaned_text.split()
                    token_counts = Counter(tokens)
                    most_common, freq = token_counts.most_common(1)[0]
                    if freq > 3 and len(most_common) <= 5:
                        continue  

                if cleaned_text:
                    blocks.append({
                        "text": cleaned_text,
                        "font_size": sum(chunk["font_sizes"]) / len(chunk["font_sizes"]) if chunk["font_sizes"] else 0,
                        "bbox": block["bbox"],
                        "page_num": page_num,
                        "is_bold": chunk["is_bold"]
                    })

    return blocks

def filter_headers_and_footers(blocks, doc):
    fb = []
    for b in blocks:
        h = doc[b["page_num"]].rect.height
        if b["page_num"] == 0 or (h*0.05) < b["bbox"][1] < (h*0.95):
            fb.append(b)
    return fb

def filter_by_score(blocks, min_score=2):
    has_any_bold = any(b['is_bold'] for b in blocks)
    filtered = []
    all_font_sizes = [b['font_size'] for b in blocks if b['font_size'] > 0]
    median_font_size = np.median(all_font_sizes) if all_font_sizes else 12.0

    for i, block in enumerate(blocks):
        text = block["text"].strip()

        if len(text) == 1:
            continue
        if re.match(r'^\d', text):
            continue
        if re.fullmatch(r'\W', text):
            continue

        score = 0
        word_count = len(text.split())

        if block['is_bold'] or not has_any_bold:
            score += 2
        if block["font_size"] >= median_font_size * 1.2:
            score += 2
        space_before = block['bbox'][1] - blocks[i-1]['bbox'][3] if i > 0 else 100
        space_after = blocks[i+1]['bbox'][1] - block['bbox'][3] if i < len(blocks) - 1 else 100
        if space_before > block['font_size'] * 0.8 and space_after > block['font_size'] * 0.5:
            score += 1
        if word_count < 10:
            score += 1
        if re.match(r'^((\d+\.)+|[A-Za-z]\.)', text):
            score += 1
        if score >= min_score and word_count < 10:
            block["score"] = score
            filtered.append(block)

    return filtered

LABEL_PROTOTYPES = {
    "H1": (
        "Main Section Title\n"
        "Chapter Title\n"
        "Major Topic Introduction\n"
        "Document Overview\n"
        "Key Area Focus\n"
        "Prominent Headline\n"
        "Executive Summary\n"
        "Primary Section\n"
        "Top-Level Category\n"
        "Overall Goals\n"
        "Final Conclusion\n"
        "General Overview\n"
        "Project Description\n"
        "Introduction"
    ),
    "H2": (
        "Subsection Title\n"
        "Specific Aspect of Main Section\n"
        "Detailed Category\n"
        "Supporting Topic\n"
        "Breakdown of Introduction/Overview\n"
        "Summary\n"
        "Background Information\n"
        "Methodology\n"
        "System Design\n"
        "Approach Explanation\n"
        "Key Features\n"
        "Important Considerations\n"
        "Findings and Observations\n"
        "Data Analysis\n"
        "Proposal Strategy"
    ),
    "H3": (
        "Specific Point or Detail\n"
        "Sub-point within a Subsection\n"
        "Example\n"
        "Key Characteristic\n"
        "Timeline\n"
        "Milestones\n"
        "Clarification\n"
        "Use Case\n"
        "Supporting Argument\n"
        "Experimental Note\n"
        "Test Case\n"
        "Limitation\n"
        "Commentary\n"
        "Highlight\n"
        "Figure Description"
    )
}

def classify_blocks_semantically(blocks):
    if not blocks:
        return []
    prototype_labels = list(LABEL_PROTOTYPES.keys())
    prototype_embs = F.normalize(embed(list(LABEL_PROTOTYPES.values())), p=2, dim=1)
    block_texts = [b["text"] for b in blocks]
    block_embs = F.normalize(embed(block_texts), p=2, dim=1)

    similarity = torch.matmul(block_embs, prototype_embs.T)
    best_indices = torch.argmax(similarity, dim=1)

    for i, block in enumerate(blocks):
        block["label"] = prototype_labels[best_indices[i]]
    return blocks

def find_title(blocks, doc):
    all_font_sizes = [b['font_size'] for b in blocks if b['font_size'] > 0]
    median_font_size = np.median(all_font_sizes) if all_font_sizes else 12.0
    candidates = []
    for block in blocks:
        if block["page_num"] > 1:
            continue

        text = block["text"].strip()
        text = text.strip(' \t\n\r"\'“‘’')
        words = text.split()
        if not (1 <= len(words) <= 15):
            continue
        if text[-1] in ".:;":
            continue
        if text[0].islower():
            continue
        if re.fullmatch(r"(?i)(introduction|table of contents|overview)", text):
            continue

        y_pos = block["bbox"][1]
        page_height = doc[block["page_num"]].rect.height
        topness = 1.0 - (y_pos / page_height)

        score = 0
        if block["page_num"] == 0:
            score += 2
        if topness > 0.6:
            score += 2
        if re.search(r"\b(guide|report|summary|overview|application|research)\b", text, re.IGNORECASE):
            score += 2
        if 3 <= len(words) <= 10:
            score += 1
        if text[0].isupper():
            score += 1
        if block["font_size"] > median_font_size:
            score += 2

        candidates.append((score, block))

    if not candidates:
        return None

    candidates.sort(key=lambda x: -x[0])
    return candidates[0][1]["text"]


def build_outline(classified, all_blocks, document_name):
    outline = []
    for block in sorted(classified, key=lambda b: (b["page_num"], b["bbox"][1])):
        if block["label"] in {"H1", "H2", "H3"} and not block["text"][0].islower():
            outline.append({
                "level": block["label"],
                "text": block["text"],
                "page": block["page_num"] ,
            })
            
    title = find_title(all_blocks, doc) or os.path.splitext(document_name)[0]
    return {
        "title": title,
        "outline": outline
    }

def assign_heading_levels_by_font(scored_blocks, fallback_to_model=True):
    if not scored_blocks:
        return []

    for block in scored_blocks:
        block["rounded_font"] = round(block["font_size"], 1)

    font_counts = Counter(b["rounded_font"] for b in scored_blocks)
    unique_fonts = sorted(font_counts.keys(), reverse=True)

    dominant_font = unique_fonts[0]
    dominant_ratio = font_counts[dominant_font] / len(scored_blocks)

    rounded_fonts = [round(f, 1) for f in unique_fonts]
    font_spread = max(rounded_fonts) - min(rounded_fonts)

    should_fallback = (
        len(unique_fonts) < 2 or
        dominant_ratio > 0.8 or
        font_spread < 1.0
    )
    if should_fallback and fallback_to_model:
        return classify_blocks_semantically([b.copy() for b in scored_blocks])

    font_to_level = {}
    if len(unique_fonts) >= 1:
        font_to_level[unique_fonts[0]] = "H1"
    if len(unique_fonts) >= 2:
        font_to_level[unique_fonts[1]] = "H2"
    if len(unique_fonts) >= 3:
        font_to_level[unique_fonts[2]] = "H3"

    for block in scored_blocks:
        block["label"] = font_to_level.get(block["rounded_font"], "Body")

    return scored_blocks

def remove_toc_blocks(blocks):
    return [b for b in blocks if b['page_num'] != 3 and not re.fullmatch(r"[. ]{5,}", b['text'])]

def override_label_by_numbering(block, median_font_size):
    text = block["text"].strip()
    font_size = block.get("font_size", 0)

    if font_size < median_font_size:
        return block  

    if re.match(r"^\d+\.\d+\.\d+", text):
        block["label"] = "H3"
    elif re.match(r"^\d+\.\d+", text):
        block["label"] = "H2"
    elif re.match(r"^\d+\.", text):
        block["label"] = "H1"

    return block

if __name__ == "__main__":

    input_dir  = Path("/app/input")
    output_dir = Path("/app/output")
    output_dir.mkdir(parents=True, exist_ok=True)

    for PDF_PATH in input_dir.glob("*.pdf"):
        MIN_SCORE = 2
        doc = fitz.open(PDF_PATH)
        all_blocks = extract_blocks(doc)
        filtered_blocks = filter_headers_and_footers(all_blocks, doc)
        filtered_blocks = remove_toc_blocks(filtered_blocks)
        scored_blocks = filter_by_score(filtered_blocks, min_score=MIN_SCORE)
        all_font_sizes = [b["font_size"] for b in scored_blocks if b["font_size"] > 0]
        median_font_size = np.median(all_font_sizes) if all_font_sizes else 12.0
        high_score_blocks = [b for b in scored_blocks if b["score"] >= MIN_SCORE]
        classified_blocks = assign_heading_levels_by_font(high_score_blocks)
        if not any(b["label"] in {"H1", "H2", "H3"} for b in classified_blocks):
            classified_blocks = classify_blocks_semantically(high_score_blocks)

        classified_blocks = [override_label_by_numbering(b, median_font_size) for b in classified_blocks]
        outline = build_outline(classified_blocks, all_blocks, os.path.basename(PDF_PATH))
        output_file = output_dir / f"{PDF_PATH.stem}.json"

        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(outline, f, indent=2, ensure_ascii=False)