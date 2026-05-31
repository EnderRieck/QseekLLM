# GitHub Setup Guide

## 1. Create GitHub Repository

1. Go to https://github.com/new
2. Repository name: `llmTrain` (or your preferred name)
3. Description: "Scalable framework for LLM pretraining with FSDP/DDP"
4. Choose: **Private** (recommended) or Public
5. **Do NOT** initialize with README, .gitignore, or license (we already have these)
6. Click "Create repository"

## 2. Push to GitHub

After creating the repository, run these commands:

```bash
# Add remote (replace YOUR_USERNAME with your GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/llmTrain.git

# Push to GitHub
git branch -M main
git push -u origin main
```

## 3. Invite Collaborators

1. Go to your repository on GitHub
2. Click "Settings" → "Collaborators"
3. Click "Add people"
4. Enter your teammate's GitHub username
5. They will receive an invitation email

## 4. Setup for Teammates

Your teammates should:

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/llmTrain.git
cd llmTrain

# Install dependencies
pip install -e '.[eval]'

# Configure environment
cp .env.example .env
# Edit .env and set DATA_DIR=/path/to/their/data

# Verify setup
python -c "from llmtrain.utils.config import load_config; print('✓ Setup OK')"
```

## 5. Collaboration Workflow

### Creating a feature branch

```bash
git checkout -b feature/your-feature-name
# Make changes
git add .
git commit -m "feat: description of changes"
git push origin feature/your-feature-name
```

### Creating a Pull Request

1. Go to your repository on GitHub
2. Click "Pull requests" → "New pull request"
3. Select your feature branch
4. Add description and click "Create pull request"
5. Request review from teammates

### Syncing with main

```bash
git checkout main
git pull origin main
git checkout your-branch
git merge main
```

## 6. Important Notes

- **Never commit** `.env` file (contains local paths)
- **Never commit** `runs/` directory (contains checkpoints, too large)
- **Never commit** data files or model checkpoints
- Use `--override` for local path customization instead of editing configs

## 7. Recommended GitHub Settings

### Branch Protection (Settings → Branches)

1. Add rule for `main` branch
2. Enable:
   - ✅ Require pull request reviews before merging
   - ✅ Require status checks to pass (if you add CI)
   - ✅ Require branches to be up to date

### .gitattributes (optional, for better diffs)

Create `.gitattributes`:
```
*.yaml linguist-language=YAML
*.jsonl linguist-generated=true
*.json linguist-generated=true
```

## 8. Next Steps

- Add CI/CD with GitHub Actions (optional)
- Set up issue templates
- Add project board for task tracking
- Configure GitHub Discussions for Q&A

## Troubleshooting

### Authentication Issues

If you get authentication errors:

```bash
# Use SSH instead of HTTPS
git remote set-url origin git@github.com:YOUR_USERNAME/llmTrain.git

# Or use GitHub CLI
gh auth login
```

### Large File Warnings

If you accidentally staged large files:

```bash
git reset HEAD path/to/large/file
git rm --cached path/to/large/file
```

Then add the path to `.gitignore`.
