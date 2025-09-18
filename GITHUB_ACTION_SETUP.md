# GitHub Action Setup Guide

## Overview
This guide will help you set up the GitHub Action to automatically run the UN scraper weekly.

## Prerequisites
- GitHub repository with the scraper code
- Supabase project with the `scraper_logs` table created
- All necessary environment variables

## Setup Steps

### 1. Push Code to GitHub
Make sure all your code is committed and pushed to your GitHub repository:
```bash
git add .
git commit -m "Add GitHub Action for weekly scraper"
git push origin main
```

### 2. Set Up Repository Secrets
Go to your GitHub repository → Settings → Secrets and variables → Actions

Add these secrets:
- `SUPABASE_URL`: `https://gjakiqtayqltssvbzasd.supabase.co`
- `SUPABASE_KEY`: `eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImdqYWtpcXRheXFsdHNzdmJ6YXNkIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0MjkxOTU4OCwiZXhwIjoyMDU4NDk1NTg4fQ.wY8akPd9J-aRVQAOwTiFuOPxWM90fvkvXpyEfPogyfw`
- `API_KEY`: `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`

### 3. Test the Workflow
1. Go to the Actions tab in your GitHub repository
2. Find "Weekly UN Scraper" workflow
3. Click "Run workflow" to test it manually
4. Monitor the execution and check the logs

### 4. Monitor Scraper Runs
After each run, you can check the results:
- **GitHub Actions**: View execution logs and status
- **Supabase Dashboard**: Check the `scraper_logs` table for detailed metrics
- **Local Script**: Run `python check_logs.py` to see recent runs

## Workflow Schedule
- **Automatic**: Every 6 days at 2:00 AM UTC
- **Manual**: Can be triggered anytime from the Actions tab

## Troubleshooting
If the workflow fails:
1. Check the Actions logs for error details
2. Verify all secrets are correctly set
3. Ensure the Supabase tables exist
4. Check that all dependencies are properly installed

## Files Created
- `.github/workflows/weekly-scraper.yml` - GitHub Action workflow
- `requirements.txt` - Python dependencies
- `GITHUB_ACTION_SETUP.md` - This setup guide