#!/bin/bash
XDG_CACHE_HOME=$PWD/.cache snakemake --cores all --use-conda --rerun-incomplete --conda-prefix $HOME/.snakemake_conda_store