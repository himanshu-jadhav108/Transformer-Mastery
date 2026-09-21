# Getting Started

This guide sets up the Transformer Mastery Course on Windows using PowerShell.
The course is designed for Python 3.9 or newer and takes approximately 40-50
hours when completed with the exercises and projects.

## 1. Install The Prerequisites

Install the following before opening the course:

- Python 3.9 or newer from [python.org](https://www.python.org/downloads/).
- Git, if you want to clone or update the repository.
- Visual Studio Code with the Python extension (recommended).

During Python installation, enable **Add Python to PATH**. Open a new
PowerShell window afterward and confirm the launcher works:

```powershell
py --version
```

## 2. Open The Repository

Change to the repository directory. Use your own path if the repository is in a
different location:

```powershell
Set-Location "D:\Learnings\transformer-mastery-course"
```

You can also open the folder in VS Code:

```powershell
code .
```

## 3. Create And Activate A Virtual Environment

Create a local environment named `.venv`:

```powershell
py -m venv .venv
```

Activate it in the current PowerShell session:

```powershell
.\.venv\Scripts\Activate.ps1
```

Your prompt should now begin with `(.venv)`. If PowerShell reports that script
execution is disabled, allow scripts for this session only:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 4. Install Dependencies

Upgrade `pip` and install the course dependencies into the active environment:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Verify the installation:

```powershell
python -c "import torch, numpy, matplotlib; print(torch.__version__); print(numpy.__version__); print(matplotlib.__version__)"
```

## 5. Run The Examples

Run commands from the repository root. Code filenames are numbered within each
code directory so the local order is explicit:

```powershell
python 02-attention\code\01-attention_manual_math.py
python 02-attention\code\02-attention_numpy.py
python 02-attention\code\visualizations\01-attention_heatmap.py
python 03-transformer-core\code\01-token_embeddings.py
python 06-training\code\debugging\01-shape_errors.py
python 07-modern-transformers\code\01-mini_bert.py
python 08-advanced-concepts\code\generation\01-greedy_decoding.py
```

To see the available scripts in the current module:

```powershell
Get-ChildItem .\02-attention\code -File -Filter "*.py" | Sort-Object Name
```

Most files are standalone demonstrations. Read the matching Markdown lesson
first, then run the scripts in order and modify the inputs to test your own
understanding.

## 6. Follow A Learning Pace

Choose a pace that leaves time to read, run, and reimplement the ideas:

| Pace | Weekly commitment | Approximate duration |
| --- | --- | --- |
| Casual | 5-6 hours | 8 weeks |
| Focused | 10+ hours | 4 weeks |
| Intensive | 15+ hours | 3 weeks |

Do not skip `01-foundations/` or `04-mathematics/` if this is your first deep
study of Transformers. They explain the recurrent-model limitations and tensor
operations that make later implementation details easier to reason about.

## 7. Track Your Progress

1. Open `00-roadmap/learning-roadmap.md`.
2. Mark each lesson after you can explain its main idea without notes.
3. Mark a code exercise after you have run it and inspected its tensor shapes.
4. Record questions and experiments beside the relevant lesson.
5. Complete the four projects in `09-projects/` before using the final checklist in `10-mastery/`.

The recommended order is `00` through `10`. Within a populated `code/`
directory, execute `01-...`, `02-...`, and so on. `05-build-from-scratch/code/`
is currently empty because that module maps back to implementation code from
modules `02` through `04`.

## 8. Troubleshooting

### Python Is Not Recognized

Install Python 3.9+, enable **Add Python to PATH**, restart PowerShell, and
run `py --version` again. The Windows Python launcher is preferred when more
than one Python installation exists.

### Virtual Environment Will Not Activate

Run the session-only policy change, then activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### A Package Import Fails

Check that `(.venv)` is visible in the prompt, then reinstall through the same
interpreter that runs the scripts:

```powershell
python -m pip install -r requirements.txt
python -c "import torch, numpy, matplotlib; print('Dependencies are available.')"
```

### A Script Path Does Not Exist

Run `Get-ChildItem -Recurse -Filter "*.py"` and use the numbered path shown by
PowerShell. For example, `self_attention.py` is now
`02-attention\code\05-self_attention.py`.

### Tensor Shape Or Mask Errors

Review `04-mathematics/code/` for shape and broadcasting exercises. The
diagnostic examples in `06-training\code\debugging\` cover shape errors,
masking errors, and NaN loss. Compare the expected `(B, T, C)` convention with
the actual shape printed by the script.

### Matplotlib Does Not Display A Figure

Run the visualization from the active virtual environment and confirm that
Matplotlib imports successfully. In a non-interactive session, save the figure
to a file or run the example inside a VS Code Python terminal.

## 9. Deactivate The Environment

When you finish a study session:

```powershell
deactivate
```

Reactivate it later from the repository root with
`.\.venv\Scripts\Activate.ps1`.

## Maintainer

**Himanshu Jadhav**  
Artificial Intelligence & Data Science Engineer

Passionate about computer vision, real-world localized model deployment, and
high-performance pipeline architecture.

- GitHub: [himanshu-jadhav108](https://github.com/himanshu-jadhav108)
- LinkedIn: [Himanshu Jadhav](https://www.linkedin.com/in/himanshu-jadhav-328082339)
- Portfolio: [himanshu-jadhav-portfolio.vercel.app](https://himanshu-jadhav-portfolio.vercel.app/)
- Instagram: [@himanshu_jadhav_108](https://www.instagram.com/himanshu_jadhav_108)
