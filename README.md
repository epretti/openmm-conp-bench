# OpenMM constant potential benchmarking

Rudimentary benchmark script for the new constant potential feature coming to
OpenMM.

```
usage: conp_benchmark.py [-h] [--platform {Reference,CPU,OpenCL}]
                         --method {matrix,cg} [--time TIME] [--repeat REPEAT]
                         [--all] [--base] [--unfrozen] [--zero] [--double]
                         [--short] [--long] [--electrolyte] [--electrode]
                         [--small] [--large]

options:
  -h, --help            show this help message and exit
  --platform {Reference,CPU,OpenCL}
                        OpenMM platform name to use
  --method {matrix,cg}  solver to run
  --time TIME           approximate time to run per measurement
  --repeat REPEAT       number of measurements per test
  --all                 run all benchmarks (may take several minutes with
                        default settings)
  --base                run 'base' benchmark
  --unfrozen            run 'unfrozen' benchmark
  --zero                run 'zero' benchmark
  --double              run 'double' benchmark
  --short               run 'short' benchmark
  --long                run 'long' benchmark
  --electrolyte         run 'electrolyte' benchmark
  --electrode           run 'electrode' benchmark
  --small               run 'small' benchmark
  --large               run 'large' benchmark
```
