# Publishing this project to GitHub

**Status:** Repository content is prepared locally, but **the project has not been confirmed as published to a GitHub account**. Do not cite a GitHub URL or claim public release until you verify it.

## Recommended repository name

`carsel-cvd-research-framework`

## If using Git locally

1. Create a new empty **public** repository under your own GitHub account (do not initialize it with an auto-generated README).
2. From this extracted project folder, run:

```bash
git init -b main
git add .
git commit -m "Release CaRSEL-CVD v0.2.0 research framework"
git remote add origin https://github.com/YOUR_USERNAME/carsel-cvd-research-framework.git
git push -u origin main
git tag v0.2.0
git push origin v0.2.0
```

3. Check the **Actions** tab and review the automated tests. The repository contains `.github/workflows/ci.yml`.
4. Create a tagged GitHub Release and, if desired, archive it in a persistent scholarly repository to obtain an immutable DOI.
5. Insert the real URL, tag, DOI, publication authorship and license details in the SoftwareX code metadata table and `README.md`.

**Privacy:** Never commit health records, private CSVs, environment secrets or model artifacts. The `artifacts/` directory and raw CSV files are gitignored by default. Review the tracked file list before each push (`git status`, `git diff --cached --stat`).
