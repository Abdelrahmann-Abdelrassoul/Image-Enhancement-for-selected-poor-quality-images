# Image Enhancement for Poor Quality Images

## Overview

This project implements classical **image enhancement techniques** for a set of poor-quality images as part of a Computer Vision mini project.

The objective is to analyze each image individually and apply suitable enhancement methods based on the degradation type:

- Component Extraction  
- Blur Enhancement  
- Noise Removal  
- Visual Enhancement  

A single method was **not** assumed to fit all images.  
Each image was treated as a separate image processing problem.

---

# Project Structure

```
MiniProject
├─ data
│  ├─ original/
│  └─ processed/
├─ reports
│  ├─ Report.pdf
│  └─ Report.tex
├─ requirements.txt
└─ src
   ├─ componentExtraction.py
   ├─ deblurring.py
   ├─ denoising.py
   ├─ io_helpers.py
   ├─ main.py
   ├─ preprocessing.py
   ├─ utils.py
   ├─ visualEnhancement.py
   └─ visualization.py
```

---

# Requirements

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Run

```bash
python src/main.py
```

---

# Output

Processed outputs are saved in:

```text
data/processed/
```

Including:

- Intermediate outputs  
- Final outputs  
- Comparison visualizations  

---

# Report

Full project report:

```text
reports/Report.pdf
```

Includes:

- Techniques used  
- Python code  
- Intermediate outputs  
- Result discussion  
- Failure analysis  
- Future work suggestions  

---

# Limitations

Known limitations:

- COVID chart blue boxes were not fully extracted.
- Classical deblurring cannot fully restore severely blurred images.
- Very small newspaper text remains challenging under heavy vertical noise.

---

# Future Improvements

Potential improvements:

- Blind deconvolution  
- OCR-aware document enhancement  
- Deep-learning image restoration  
- Adaptive color segmentation  
- Learned denoisers

---
