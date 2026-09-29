---
title: Production ASR deployment with FunASR
author:
  name: "FunASR contributors"
date: 2026-08-13
---

# Production ASR deployment with FunASR

ESPnet and ESPnet3 provide research workflows for training, evaluating, and
publishing speech models. When an application also needs a ready-made ASR
serving path, [FunASR](https://github.com/modelscope/FunASR) can be used as an
optional, external production backend.

This is an interoperability pattern, not a direct model conversion path.
ESPnet checkpoints do not automatically load in FunASR, and adding FunASR does
not change an ESPnet recipe or its dependencies.

## When this pattern fits

Keep ESPnet in the research and evaluation loop, and consider FunASR for an
application-facing service when you need one or more of these capabilities:

- an OpenAI-compatible transcription API
- Docker or Kubernetes deployment
- streaming and offline ASR runtimes
- FunASR models such as Paraformer, SenseVoice, or Fun-ASR-Nano
- a vLLM path for supported speech language models
- native `llama.cpp`-based deployment for supported models and platforms

Compare the available paths in the
[FunASR deployment matrix](https://github.com/modelscope/FunASR/blob/main/docs/deployment_matrix.md).

## Service boundary

A simple architecture keeps the two toolkits independent:

```text
ESPnet / ESPnet3
  train -> evaluate -> select a research checkpoint

Application
  audio -> FunASR service -> transcript -> downstream processing
```

This boundary lets research code continue to use ESPnet datasets, metrics, and
recipes while the application calls a separately versioned inference service.
Compare both paths on the same held-out audio and task-specific metrics before
replacing any existing backend.

## Start with the OpenAI-compatible API

For HTTP integration, begin with FunASR's
[OpenAI-compatible API example](https://github.com/modelscope/FunASR/tree/main/examples/openai_api).
It includes a server, smoke tests, Docker Compose, client examples, an OpenAPI
document, and Kubernetes manifests.

The service exposes a transcription boundary that can be called independently
from ESPnet code. Pin the FunASR and model revisions in deployment, then record
those revisions with evaluation results so experiments remain reproducible.

## Other production runtimes

Choose a runtime based on the model and operating environment rather than
assuming every model supports every backend:

- Use the [vLLM guide](https://github.com/modelscope/FunASR/blob/main/docs/vllm_guide.md)
  for supported speech language models and GPU serving.
- Use the [FunASR runtime documentation](https://github.com/modelscope/FunASR/tree/main/runtime)
  for native streaming, offline, mobile, and platform-specific options.
- Use FunASR's published release assets when a supported prebuilt
  `llama.cpp` runtime matches the target platform.

::: important
Check the deployment matrix and the selected model documentation before
choosing a backend. Runtime support, streaming behavior, timestamps, and
language coverage differ by model.
:::

## Production checklist

Before routing application traffic to the service:

1. Evaluate representative audio with the same text normalization and metrics
   used by the ESPnet experiment.
2. Test long audio, silence, malformed files, and concurrent requests.
3. Pin package, container, runtime, and model revisions.
4. Set request-size, duration, timeout, and concurrency limits.
5. Monitor latency, error rate, GPU or CPU memory, and transcript quality.
6. Keep a rollback path to the previous model and service revision.

Treat the serving backend as a separately tested production component. This
keeps the ESPnet research workflow reproducible and makes deployment changes
independent from model-development changes.
