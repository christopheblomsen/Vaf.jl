# Vaf

[![Build Status](https://github.com/christopheblomsen/Vaf.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/christopheblomsen/Vaf.jl/actions/workflows/CI.yml?query=branch%3Amain)

## Vaftrundir

**Vafthrundir**, or Vaf, is a fork of [**μspel**](https://github.com/tiagopereira/Muspel.jl/tree/main) for radiative transfer in stellar atmospheres. Aiming to solve it with GPU paralization. It is named after Vafthrundir, the mighty weaver, that battled Odin in the contest of wits. Vaf means *to weave or entagle*.

μspel is still under development, and the API is subject to change. Documentation is currently limited to that generated from docstrings.

## Installation
Enter a Julia environment in the parent directory.

```Julia
pkg> dev ./Vaf.jl
```

## Simulations
The scripts to run the benchmark and simulations can be found [here](https://github.com/christopheblomsen/Vaf.jl/tree/main/scripts). Where the [intensity calculations](https://github.com/christopheblomsen/Vaf.jl/tree/main/scripts/intensity_speedup.ipynb) and the [angle-average intensity](https://github.com/christopheblomsen/Vaf.jl/tree/main/scripts/mean_intensity_speedup.ipynb) are seperated. 
