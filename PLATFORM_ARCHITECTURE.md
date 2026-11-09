# Leo platform and hardware scalability

## Non-negotiable design principle

Leo's model architecture and research roadmap must not be capped by the owner's current 4 GB RAM Android phone. That phone is an early deployment and usability test target only. It is not the training machine and does not define the maximum model size.

## One project, versioned model family

Maintain a common model specification, checkpoint metadata, tokenizer version, and inference contract. Produce different model sizes/quantizations from Leo's own trained checkpoints when technically supported. Small builds are deployment variants, not separate unrelated AI identities.

Every checkpoint should record:
- Architecture and configuration version.
- Parameter count and context length.
- Tokenizer version and vocabulary/tokenizer provenance.
- Training data manifest/version and language mixture.
- Training seed, step count, optimizer/schedule, and evaluation results.
- Export/quantization method and runtime compatibility.

Do not promise that every checkpoint is interchangeable. Define explicit compatibility and migration rules and test them.

## Target platforms

### Android phones and tablets
Start with a small quantized inference target and conservative context/memory settings. Benchmark RAM, storage, latency, battery, thermal behavior, and crash recovery on actual devices. A 4 GB device may only run a very small model comfortably; capability will be limited by that deployment profile.

### Higher-memory Android devices
Allow larger variants and longer context only after device-specific profiling. Do not hard-code the 4 GB profile into the core model or training configuration.

### Windows, Linux, and macOS computers
Support CPU and accelerator-backed inference where available. Desktop builds can expose larger models, longer context, local document retrieval, coding workspaces, and app-build tools, subject to permissions and installed toolchains.

## Shared capability layers

Keep the model core independent from optional systems:
- Retrieval/document indexing.
- Math and data-analysis tools.
- Sandboxed code execution and test/build orchestration for general software projects.
- Speech, image, and other modalities as separate modules.
- Permissioned connectors and internet access.

Tool use must be explicit, auditable, and sandboxed. Do not imply that the base model itself can execute actions without these systems.

## Free-first development

Prefer open-source tools and free resources where available. Do not make paid model APIs or proprietary pretrained weights required dependencies. Free GPU sessions are limited, may disconnect, and are not a reliable substitute for a funded training cluster. Save resumable checkpoints and publish exact compute assumptions. Never claim that large-scale training is free or guaranteed.

## Platform roadmap

1. Keep model research and training platform-neutral.
2. Define a stable checkpoint and inference interface.
3. Build and test a desktop reference runtime.
4. Add an Android inference build after model export/runtime compatibility is demonstrated.
5. Add quantized model variants and device profiles.
6. Add permission-controlled coding assistance and general software build/test workflows. Building AI-powered applications is explicitly out of scope.
