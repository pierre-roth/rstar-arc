# Sources and licenses

Original rStar-ARC code is [MIT-licensed](../LICENSE).

- [rStar-Math](https://github.com/microsoft/rStar) provided the search and
  self-evolution methodology.
- [ARC-AGI](https://github.com/fchollet/ARC-AGI) provides the benchmark and sample
  puzzle grids. Its [Apache 2.0 license](licenses/ARC-Apache-2.0.txt) is included.
- [`rstar_deepthink/vendor/rearc/`](../rstar_deepthink/vendor/rearc/) contains
  code adapted from Michael Hodel's [RE-ARC](https://github.com/michaelhodel/re-arc)
  and [ARC-DSL](https://github.com/michaelhodel/arc-dsl).
  Their MIT notices are included in [licenses/](licenses/).
- [`rstar_deepthink/vendor/barc.py`](../rstar_deepthink/vendor/barc.py) is adapted
  from [BARC's grid helpers](https://github.com/xu3kev/BARC/blob/master/seeds/common.py).
  BARC does not provide a top-level license for this file; rStar-ARC's MIT grant
  does not cover it.

`data/examples/solutions.jsonl` contains curated prompt examples.
`data/bootstrap/` contains processed HARC and LARC descriptions; their
preparation is recorded in the [research notes](research-notes.md).

The conversion scripts use the
[BARC synthetic dataset](https://huggingface.co/datasets/barc0/200k_HEAVY_gpt4o-description-gpt4omini-code_generated_problems).
External datasets and model weights retain their source terms.
