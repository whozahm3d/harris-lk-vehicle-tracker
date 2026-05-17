# 🤝 Contributing to Harris–LK Vehicle Tracker

Thank you for your interest in contributing to this project. Contributions are welcome in the form of bug reports, feature suggestions, documentation improvements, and code enhancements.

---

## 📋 Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Report a Bug](#how-to-report-a-bug)
- [How to Suggest a Feature](#how-to-suggest-a-feature)
- [How to Submit a Pull Request](#how-to-submit-a-pull-request)
- [Development Setup](#development-setup)
- [Code Style Guidelines](#code-style-guidelines)

---

## Code of Conduct

Please be respectful and constructive in all interactions. This is an academic project and contributions should align with its classical CV focus — no deep learning dependencies will be accepted.

---

## How to Report a Bug

1. Check the [Troubleshooting section](README.md#-troubleshooting) in the README first.
2. Search existing [Issues](https://github.com/whozahm3d/harris-lk-vehicle-tracker/issues) to see if it has already been reported.
3. If not, open a new issue and include:
   - A clear title describing the problem
   - Steps to reproduce the issue
   - Your operating system and Python version
   - The full error message and traceback
   - Your `config.yaml` contents (remove any sensitive file paths)

---

## How to Suggest a Feature

Open a new issue with the label `enhancement` and include:
- A clear description of the proposed feature
- Why it would be useful within the scope of classical CV tracking
- Any relevant references or prior work

> **Note:** Features that introduce deep learning components, external model weights, or GPU-only dependencies are outside the scope of this project and will not be accepted.

---

## How to Submit a Pull Request

1. Fork the repository
2. Create a new branch from `main`:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes — keep commits small and focused
4. Test your changes on at least one video before submitting
5. Update `README.md` or docstrings if your change affects usage or parameters
6. Open a pull request against `main` with a clear description of what you changed and why

---

## Development Setup

```bash
git clone https://github.com/whozahm3d/harris-lk-vehicle-tracker.git
cd harris-lk-vehicle-tracker
python -m venv venv
venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

---

## Code Style Guidelines

- Follow the existing module structure — new functionality belongs in the appropriate `core/`, `utils/`, `metrics/`, or `visualization/` module
- All parameters must be read from `config.yaml` via `ConfigLoader` — no hardcoded values
- Use type hints and docstrings consistent with the existing codebase style
- Keep functions focused and single-purpose
- Log meaningful messages using `AppLogger` at the appropriate level (INFO / DEBUG / WARNING)

---

<p align="center">Thank you for helping improve this project ❤️</p>
