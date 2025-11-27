# RTpipeline Documentation Index

This directory contains technical documentation, advanced guides, and development resources for RTpipeline.

## 📚 **Documentation Overview**

### For End Users
👉 **Start with the [main README](../README.md)** for quick start and basic usage.

---

## 📖 **User Guides** (Root Directory)

| Document | Description | Audience |
|----------|-------------|----------|
| [**GETTING_STARTED.md**](../GETTING_STARTED.md) | Complete beginner's guide | New users |
| [**WEBUI.md**](../WEBUI.md) | Web UI documentation | All users |
| [**output_format_quick_ref.md**](../output_format_quick_ref.md) | Output format reference with code snippets | Data analysts |
| [**setup_new_project.sh**](../setup_new_project.sh) | Interactive setup for native installs | Local installation users |
| [**setup_docker_project.sh**](../setup_docker_project.sh) | **NEW:** Interactive Docker project setup | Docker users |

---

## 🔧 **Technical Documentation** (This Directory)

### Architecture & Design

| Document | Description |
|----------|-------------|
| [**PIPELINE_ARCHITECTURE.md**](PIPELINE_ARCHITECTURE.md) | Pipeline architecture and design decisions |
| [**PARALLELIZATION.md**](PARALLELIZATION.md) | Parallelization strategies and performance tuning |

### Deployment & Operations

| Document | Description |
|----------|-------------|
| [**DOCKER.md**](DOCKER.md) | Docker/Singularity deployment and technical compatibility |
| [**DOCKER_SETUP_GUIDE.md**](DOCKER_SETUP_GUIDE.md) | **NEW:** Comprehensive Docker setup with config guide and examples |
| [**SECURITY.md**](SECURITY.md) | Security guide for production deployments |
| [**TROUBLESHOOTING.md**](TROUBLESHOOTING.md) | Troubleshooting hang issues and timeouts |

### Advanced Features

| Document | Description |
|----------|-------------|
| [**custom_models.md**](custom_models.md) | Using custom nnUNet segmentation models |
| [**RADIOMICS_ROBUSTNESS.md**](RADIOMICS_ROBUSTNESS.md) | Radiomics robustness workflow, configuration, and research references |
| [**SYSTEMATIC_CT_CROPPING.md**](SYSTEMATIC_CT_CROPPING.md) | Systematic anatomical cropping guide |

### Data & Quality Control

| Document | Description |
|----------|-------------|
| [**Guide to Results Interpretation.md**](Guide%20to%20Results%20Interpretation.md) | Interpreting pipeline results |
| [**qc_cropping_audit.md**](qc_cropping_audit.md) | CT cropping quality control |

---

## 🎯 **Quick Navigation by Use Case**

### "I want to understand the pipeline architecture"
→ [PIPELINE_ARCHITECTURE.md](PIPELINE_ARCHITECTURE.md)

### "I need to optimize performance"
→ [PARALLELIZATION.md](PARALLELIZATION.md)

### "I'm deploying with Docker"
→ [DOCKER_SETUP_GUIDE.md](DOCKER_SETUP_GUIDE.md) (comprehensive guide with examples)
→ [DOCKER.md](DOCKER.md) (technical compatibility guide)

### "The pipeline is hanging or timing out"
→ [TROUBLESHOOTING.md](TROUBLESHOOTING.md)

### "I want to use my own segmentation models"
→ [custom_models.md](custom_models.md)

### "I need to interpret my results"
→ [Guide to Results Interpretation.md](Guide%20to%20Results%20Interpretation.md) and [output_format_quick_ref.md](../output_format_quick_ref.md)

### "I must validate radiomics stability before modelling"
→ [RADIOMICS_ROBUSTNESS.md](RADIOMICS_ROBUSTNESS.md)

### "I'm working with CT cropping"
→ [SYSTEMATIC_CT_CROPPING.md](SYSTEMATIC_CT_CROPPING.md)

### "I'm deploying in production"
→ [SECURITY.md](SECURITY.md)

---

## 📦 **Directory Structure**

```
rtpipeline/
│
├── README.md                          ← Start here
├── GETTING_STARTED.md                 ← Beginner's guide
├── WEBUI.md                           ← Web UI guide
├── output_format_quick_ref.md         ← Output format reference
├── setup_new_project.sh               ← Interactive native setup
├── setup_docker_project.sh            ← Interactive Docker setup
│
├── docs/                              ← Technical documentation (you are here)
│   ├── README.md                      ← This file
│   ├── PIPELINE_ARCHITECTURE.md       ← Architecture overview
│   ├── PARALLELIZATION.md             ← Performance tuning
│   ├── DOCKER.md                      ← Docker technical guide
│   ├── DOCKER_SETUP_GUIDE.md          ← Docker setup & config guide
│   ├── TROUBLESHOOTING.md             ← Debugging guide
│   ├── SECURITY.md                    ← Production security
│   ├── custom_models.md               ← Custom models
│   ├── RADIOMICS_ROBUSTNESS.md        ← Robustness analysis
│   ├── SYSTEMATIC_CT_CROPPING.md      ← CT cropping guide
│   ├── Guide to Results Interpretation.md
│   └── qc_cropping_audit.md
│
├── rtpipeline/                        ← Python package
├── envs/                              ← Conda environments
├── webui/                             ← Web UI application
├── scripts/                           ← Utility scripts
└── custom_models/                     ← Model repository
```

---

## 🆘 **Getting Help**

1. **Check the main [README](../README.md)** for quick start
2. **Read [GETTING_STARTED.md](../GETTING_STARTED.md)** for step-by-step guide
3. **Review [TROUBLESHOOTING.md](TROUBLESHOOTING.md)** for common issues
4. **Search the [GitHub Issues](https://github.com/kstawiski/rtpipeline/issues)** for reported problems
5. **Open a new issue** if you can't find a solution

---

## 🤝 **Contributing**

For development and contribution guidelines, see the main [README](../README.md#contributing).

---

**Last Updated:** 2025-11-19
**Pipeline Version:** v2.0+
