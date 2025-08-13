# ODR+ — Stratified BrowseComp Evaluation

**ODR+** is a drop-in enhancement for [Open Deep Research (ODR)](https://github.com/nickscamara/open-deep-research) that adds **BrowseComp-optimized evaluation** capabilities.  
It includes **subquestion generation**, **analysis & planning**, and **structured 3-line answer synthesis** for reproducible experiments.

---

## ✨ Features

- **Drop-in replacement** for ODR’s API route at `src/app/api/chat/route.ts`
- **Stratified sampling script** to create a balanced 60-question subset from BrowseComp
- **BrowseComp-focused flow** with constraint awareness and structured output
- **Reproducible evaluation** with fixed seeds and documented methodology

---

## 📂 Repository Structure

```plaintext
src/app/api/chat/route.ts       # Enhanced ODR route (drop-in replacement)
experiments/browsecomp/
├── stratified_sampler.py       # Creates 60-question stratified dataset
└── questions.json              # Exact question set used in paper
requirements.txt                # Python dependencies
.gitignore
LICENSE
README.md
