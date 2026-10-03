---
title: SupportHR Classifier
emoji: 🧭
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
short_description: CV industry classifier (24 labels) for SupportHR / CV Match
---

# SupportHR CV Industry Classifier

Deployed automatically from [TechFutureAIFPT/supporthr-ml-pipeline](https://github.com/TechFutureAIFPT/supporthr-ml-pipeline)
by GitHub Actions — do not edit files in this Space by hand, they are overwritten on every deploy.

| Endpoint | Method | Auth |
| --- | --- | --- |
| `/health` | GET | public |
| `/api/cv/classifier-status` | GET | `X-Classifier-Key` |
| `/api/cv/classify-industry` | POST `{"cv_text": "...", "top_k": 3}` | `X-Classifier-Key` |

The key is the Space secret `CLASSIFIER_API_KEY`; the main API sends it via
`LOCAL_CLASSIFIER_REMOTE_API_KEY`.
