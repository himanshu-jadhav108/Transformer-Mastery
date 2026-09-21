# Getting Started

This guide takes you from zero to running your first Transformer notebook in about 10 minutes on **Windows, macOS, or Linux**.

---

## TL;DR

Already comfortable with Python, Git, and virtual environments? Run the commands below from the directory where you keep your projects:

### macOS / Linux

```bash
git clone https://github.com/himanshu-jadhav108/Transformer-Mastery.git
cd Transformer-Mastery
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_env.py
```

### Windows (PowerShell)

```powershell
git clone https://github.com/himanshu-jadhav108/Transformer-Mastery.git
Set-Location Transformer-Mastery
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python scripts/check_env.py
```

Then open the project in **VS Code** (`code .`) and launch your first notebook:
`02-attention/Practice_02.ipynb`.

If `check_env.py` passes and the first notebook cells execute cleanly, you are fully set up!

---

## Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone the Repository](#2-clone-the-repository)
3. [Create and Activate a Virtual Environment](#3-create-and-activate-a-virtual-environment)
4. [Install Dependencies](#4-install-dependencies)
5. [Verify Your Setup with `check_env.py`](#5-verify-your-setup-with-check_envpy)
6. [How to Run the Course Notebooks](#6-how-to-run-the-course-notebooks)
7. [Your First 30 Minutes](#7-your-first-30-minutes)
8. [Module 09 Projects Notice](#8-module-09-projects-notice)
9. [Troubleshooting Guide](#9-troubleshooting-guide)
10. [Daily Study Workflow](#10-daily-study-workflow)

---

## 1. Prerequisites

| Requirement | Details |
| --- | --- |
| **Python** | 3.9 or newer (64-bit). Tested on Python 3.9 through 3.13. |
| **Git** | For cloning and tracking updates. |
| **Disk Space** | ~2–3 GB (PyTorch and scientific libraries). |
| **Hardware Accelerator** | Optional. All course models run smoothly on standard CPU. CUDA and Apple Silicon (MPS) are detected automatically if present. |
| **Recommended Editor** | [VS Code](https://code.visualstudio.com/) with the **Python** and **Jupyter** extensions. |

> **Prefer zero local setup?** Launch the repo instantly in your browser with [GitHub Codespaces](https://codespaces.new/himanshu-jadhav108/Transformer-Mastery).

---

## 2. Clone the Repository

Clone the repository and enter the root directory:

```bash
git clone https://github.com/himanshu-jadhav108/Transformer-Mastery.git
cd Transformer-Mastery
```

> **Tip:** If you plan to save notebook outputs or personal annotations, consider [forking the repository](https://github.com/himanshu-jadhav108/Transformer-Mastery/fork) first and cloning your fork.

---

## 3. Create and Activate a Virtual Environment

A virtual environment isolates course dependencies from the rest of your system.

<details open>
<summary><b>macOS / Linux</b></summary>

```bash
python3 -m venv .venv
source .venv/bin/activate
```
</details>

<details>
<summary><b>Windows PowerShell</b></summary>

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell shows a script execution error, run:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```
</details>

<details>
<summary><b>Windows Command Prompt (CMD)</b></summary>

```bat
py -m venv .venv
.venv\Scripts\activate.bat
```
</details>

When activated, your terminal prompt will be prefixed with `(.venv)`.

---

## 4. Install Dependencies

Install the core dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` includes:
- `torch` ($\ge$ 2.0)
- `numpy` ($\ge$ 1.24)
- `matplotlib` ($\ge$ 3.7)
- `ipykernel` ($\ge$ 6.20) — enables notebook execution in VS Code and Jupyter

If you plan to run notebooks directly inside a web browser, install JupyterLab:
```bash
python -m pip install notebook
```

---

## 5. Verify Your Setup with `check_env.py`

Run the included environment diagnostic from the repository root:

```bash
python scripts/check_env.py
```

A healthy result looks like this:

```text
Transformer Mastery - environment check

Python      3.11.x  (d:\...\Transformer-Mastery\.venv\Scripts\python.exe)
torch       2.x.x
numpy       x.x.x
matplotlib  x.x.x
torch sanity check passed (tiny attention computation)
Accelerator CPU only (or CUDA / Apple MPS)

All checks passed. You're ready to start.
```

If any check fails, see the [Troubleshooting Guide](#9-troubleshooting-guide).

---

## 6. How to Run the Course Notebooks

The practical implementations in this course are structured as self-contained **Jupyter Notebooks (`.ipynb`)**:

| Module | Notebook | What You Build |
| --- | --- | --- |
| **01 Foundations** | `01-foundations/Practice_01.ipynb` | Embeddings, RNN forward pass, sequence modeling |
| **02 Attention** | `02-attention/Practice_02.ipynb` | QKV projections, self-attention, attention heatmaps |
| **03 Core Blocks** | `03-transformer-core/Practice_03.ipynb` | Positional encoding, FFNs, LayerNorm & RMSNorm |
| **04 Mathematics** | `04-mathematics/Practical_04.ipynb` | Tensor reshaping, broadcasting, `einsum` operations |
| **05 Build From Scratch** | `05-build-from-scratch/GPT_From_Scratch.ipynb` | Full decoder-only GPT model from raw PyTorch |
| **06 Training** | `06-training/06-Training.ipynb` | Training loops, loss computation & debugging traps |
| **07 Modern Family** | `07-modern-transformers/Modern-Transformer-Family.ipynb` | BERT, GPT, LLaMA architectural implementations |
| **08 Advanced Concepts** | `08-advanced-concepts/Adevanced-Concepts.ipynb` | RoPE, KV cache inference, and MoE routing |

### Running in VS Code (Recommended)
1. Open the repository root: `code .`
2. Open any `.ipynb` notebook file.
3. In the upper-right corner of the notebook editor, click **Select Kernel**.
4. Choose **Python Environments...** and pick your `.venv` virtual environment.
5. Click **Run All** or execute cells individually with `Shift + Enter`.

### Running in JupyterLab / Browser
1. Launch Jupyter from your activated terminal:
   ```bash
   jupyter lab    # or: jupyter notebook
   ```
2. Navigate to any module folder and open the notebook.

---

## 7. Your First 30 Minutes

Follow this loop to absorb concepts effectively:

1. **Read (10 min):** Open `02-attention/01-attention-intuition.md` to understand why Query, Key, and Value exist.
2. **Execute (10 min):** Open `02-attention/Practice_02.ipynb`. Run through the first few cells and observe how attention weights form probability distributions across token positions.
3. **Predict Shapes (5 min):** Before executing a transformation cell, write down the expected output dimensions (e.g., `(B, h, T, d_k)`).
4. **Inspect Attention Heatmaps (5 min):** View the generated heatmaps to see which tokens attend to which tokens.

---

## 8. Module 09 Projects Notice

Module `09-projects/` contains **4 Capstone Projects**:
1. **Project 1:** Attention Visualizer
2. **Project 2:** Text Classification Transformer
3. **Project 3:** Mini Language Model
4. **Project 4:** Mini GPT from Scratch

### Current Status:
- Detailed architectural blueprints, concept explanations, and implementation roadmaps (`project-01-*.md` through `project-04-*.md`) are ready in `09-projects/`.
- Dedicated, step-by-step Jupyter Notebooks for each project are actively in development and will be added in an upcoming update. You can explore the architectural designs and begin structuring your own implementations right away.

---

## 9. Troubleshooting Guide

| Symptom | Likely Cause | Solution |
| --- | --- | --- |
| `python` or `py` is not recognized | Python is not installed or not added to system PATH | Install Python 3.9+ from [python.org](https://www.python.org/downloads/). On Windows, check **"Add Python to PATH"**. Restart your terminal. |
| PowerShell: "Script execution is disabled" | PowerShell execution policy restriction | Run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass`, then activate `.\.venv\Scripts\Activate.ps1`. |
| `ModuleNotFoundError: No module named 'torch'` | Virtual environment is not active or packages went to global Python | Verify `(.venv)` appears in your prompt. Re-run `python -m pip install -r requirements.txt`. |
| VS Code: "Select a Kernel" prompt | VS Code needs to locate your `.venv` | Click **Select Kernel** in top-right → **Python Environments** → choose `.venv`. |
| Jupyter: `ModuleNotFoundError` inside notebook | Notebook running in a different Python kernel | Ensure the notebook is running inside the `.venv` kernel (which has `ipykernel` installed). |
| Plot does not render inline in notebook | Missing inline backend directive | Notebooks include `%matplotlib inline` automatically. In standard scripts, plots save to disk or render interactively. |
| Out of Memory (OOM) | Batch size or sequence length too large | Reduce batch size (`B`) or sequence length (`T`) in the notebook configuration cell. |

---

## 10. Daily Study Workflow

Whenever you return to study:

1. **Activate your environment:**
   ```bash
   cd Transformer-Mastery
   source .venv/bin/activate       # Windows PowerShell: .\.venv\Scripts\Activate.ps1
   ```
2. **Open in VS Code or launch Jupyter:**
   ```bash
   code .                          # or: jupyter lab
   ```
3. **Deactivate when finished:**
   ```bash
   deactivate
   ```

---

## 👤 Maintainer

<br>

<p align="center">
  <table align="center" style="border: 1px solid rgba(255,255,255,0.1); border-radius: 16px; background: rgba(30, 41, 59, 0.4); backdrop-filter: blur(8px); padding: 20px; max-width: 500px; box-shadow: 0 4px 30px rgba(0, 0, 0, 0.3);">
    <tr>
      <td align="center">
        <h3 style="margin: 0; color: #38bdf8; font-size: 1.6em; font-weight: 800; letter-spacing: -0.5px;">Himanshu Jadhav</h3>
        <p style="color: #94a3b8; font-weight: 500; margin: 4px 0 15px 0;">Artificial Intelligence & Data Science Engineer</p>
        <p style="color: #cbd5e1; font-size: 0.95em; max-width: 400px; line-height: 1.5; margin-bottom: 20px;">
          Passionate about deep learning architectures, Transformer models, real-world localized model deployment, and high-performance pipeline architecture.
        </p>
        <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
          <a href="https://github.com/himanshu-jadhav108" target="_blank"><img src="https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
          <a href="https://www.linkedin.com/in/himanshu-jadhav-328082339" target="_blank"><img src="https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white" alt="LinkedIn"></a>
          <a href="https://himanshu-jadhav-portfolio.vercel.app/" target="_blank"><img src="https://img.shields.io/badge/Portfolio-FFD700?style=for-the-badge&logo=google-chrome&logoColor=black" alt="Portfolio"></a>
          <a href="https://www.instagram.com/himanshu_jadhav_108" target="_blank"><img src="https://img.shields.io/badge/Instagram-E4405F?style=for-the-badge&logo=instagram&logoColor=white" alt="Instagram"></a>
        </div>
      </td>
    </tr>
  </table>
</p>

<br>

---

<div align="center">

**If you found this course helpful, please consider giving it a ⭐!**

[Back to Top](#getting-started)

</div>

