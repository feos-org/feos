# Global thread pool

Several functions in `feos` use [Rayon](https://github.com/rayon-rs/rayon) for parallelism.
By default, Rayon uses all logical CPUs available on your machine, which is usually what you want when working on your local machine. 
In other environments, for example HPC clusters, you may want to limit the number of threads to match your job allocation.

There are three ways to configure this, in order of priority:

- `FEOS_MAX_THREADS` environment variable: for HPC or "script" environments, defined before launching Python
- `feos.set_num_threads()`:  for interactive use, at the top of a script or notebook
- Do nothing: local machines where using all cores is fine

You can get the number of threads configured via `feos.get_num_threads()`.

## Important
- The thread pool can only be configured **once** per Python process.
- Whichever method runs first wins. Any later attempt to change it will have no effect and a warning will be emitted.
- Calling `get_num_threads` without setting `FEOS_MAX_THREADS` or `set_num_threads` will initialze the thread pool with the default (all logical CPUs).
- To test the different behaviour in a notebook, you have to restart the kernel and start from the respective cell you want to test.

## Example Usage

```python
import feos

feos.set_num_threads(4)
print(f"Active threads: {feos.get_num_threads()}")
```

```{eval-rst}
.. currentmodule:: feos

.. autosummary::
    :toctree: generated/

    set_num_threads
    get_num_threads
```
