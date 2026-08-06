# Hugging Face Spaces deployment guide

> **Deploys are automated.** Every push to `main` runs `.github/workflows/deploy.yml`,
> which overlays the runtime snapshot (code + `deploy/huggingface/` files +
> `requirements.txt`) onto the Space via `deploy/huggingface/assemble.sh` and pushes
> it — preserving the Space's `models/` artefacts. A weekly `space-drift` job fails
> CI if the live Space ever stops matching `main`. The one-time setup it needs:
> a **write-scoped** HF token saved as the `HF_TOKEN` repository secret
> (Settings → Secrets and variables → Actions). The manual steps below are the
> original bootstrap procedure — still valid for creating the Space from scratch
> or for emergency pushes, but routine deploys must ride `main`, because the
> hand-deployed Space previously drifted 3 months stale while `main` carried the
> fixes.

The Space itself is created once, by hand. Everything after that rides
`main`; the steps that used to describe the hand-deploy path are gone,
because that path is what drifted.

## Prerequisites

1. A Hugging Face account. Sign up free at <https://huggingface.co/join> if you don't have one.
2. Git installed locally.

## Step 1 — Create a Hugging Face access token

Profile → **Settings → Access Tokens** → **New token**, **write** scope.
Save it in the GitHub repo as the `HF_TOKEN` secret
(Settings → Secrets and variables → Actions). `deploy.yml` reads it from there.

## Step 2 — Create the Space

<https://huggingface.co/new-space> → SDK **Streamlit**, hardware **CPU basic**,
visibility **Public**. Leave it empty; the first push from `main` fills it.

## Rollback

Each push is a Space revision. Visit the Space's **Settings → Revisions** tab to roll back to a prior commit SHA.

## Cost

Free tier (CPU basic). No charge for the demo. If you upgrade to GPU or A10G later for higher throughput, costs apply per the HF pricing page.
