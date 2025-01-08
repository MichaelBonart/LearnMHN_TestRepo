# General Rules for Working on the Package

The package is regularly updated with new features. Ideally, the following rules should be applied:

- **Backwards Compatibility**: Backwards compatibility should always be ensured. This means that results generated with older package versions must also be reproducible with the current version. Changes to the package should not break older reproduction scripts. This implies:
  - Functions, classes, modules and constants should not be renamed whenever possible.
  - Existing methods/classes that are extended with new functionalities must still work with their old invocation parameters and produce the same results. New parameters must be added as keyword arguments with default values, and the return value of a method must contain exactly the same objects for old inputs.

- **Testing**: Whenever possible, each function should have a corresponding test function located in the `test/` folder. These tests are especially important to protect against issues caused by future changes to the package. For simpler functions, such as plotting functions, it may suffice to simply execute the function in a test to ensure it runs without errors.

- **Documentation and Type Hints**: Each function must include:
  - A docstring with a brief description of the function, all input parameters, and the return value.
  - Type hints for all input and output parameters.

- **New Modules**: When creating a new module:
  - Add a brief docstring at the beginning that explains the module’s contents in broad terms.
  - Include a comment listing all authors of the module in the format: `# author(s): Max Mustermann`.

- **Documentation Updates**: Ensure that any new functionality is included in the package documentation (see `doc/` folder).

- **PEP8 Compliance**: Code should adhere to PEP8 standards. The only exception is line length, which may exceed the default limit (up to approximately 160 characters).

---

# Versioning

When working on a new version of the `mhn` package, the version in `setup.py` should always be updated first. To do so, simply adjust the `VERSION` variable.

For a version X.Y.Z, the following rules apply:

1. If there are only minimal changes, such as a bug fix, increment Z.
2. If new functionalities are added, increment Y.
3. If major changes are made that render the package no longer backwards compatible, increment X.

(All numbers to the right of the incremented number are reset to 0.) 

`setup.py` uses the `VERSION` variable to communicate the current package version to `setuptools`. Additionally, the version must be accessible within the package itself, allowing the user to query the package version via `mhn.__version__`. To achieve this, `setup.py` creates a `METADATA` file within the package, which the package can access at runtime.

---

# Dependencies

Like most Python packages, the `mhn` package requires other Python packages to function correctly. Two cases must be distinguished:

1. **Installation Dependencies**: These are required for the package installation process, particularly for compiling the Cython components (e.g., `setuptools`, `Cython`, and `numpy`).
2. **Runtime Dependencies**: These are needed at runtime, such as `pandas` and `matplotlib`.

### Installation Dependencies

Dependencies required for installation must be specified in the `pyproject.toml` file, while runtime dependencies are specified in `setup.py`. Packages required for both installation and runtime (e.g., `numpy` and `scipy`) must be declared in both `pyproject.toml` and `setup.py`.

### Versioning Challenges

Selecting the correct version for dependencies can be challenging, especially for packages like `numpy` and `scipy`:

- **Compatibility**: Only certain `scipy` and `numpy` versions are compatible with each other. Compatibility information can be found on [this page](https://docs.scipy.org/doc/scipy/dev/toolchain.html).
- **Runtime Alignment**: The runtime versions of `scipy` and `numpy` on the user’s environment must be at least as recent as the versions used during the Cython code compilation. For instance, if the code is compiled with `numpy==1.26.4`, the user must have `numpy>=1.26.4` installed.

### Installation Workflow

When `pip` installs a package:

1. It creates a virtual environment using the dependencies specified in `pyproject.toml` and compiles the Cython code.
2. It then installs the `mhn` package and the runtime dependencies listed in `setup.py` within the target environment.

This process can lead to potential issues. For example:

- A user’s environment already contains `numpy`, and it cannot be upgraded due to other dependencies (e.g., `numpy>=2.0.0` is not backwards compatible with `numpy<2.0.0`).
- If we specify only a minimum version of `numpy` for Cython compilation, `pip` would install the latest version of `numpy` during the build process. Consequently, the runtime environment would also require this latest version, potentially breaking the user's setup.

### Solution: Fixed Versions for Compilation

To maximize compatibility, fixed versions of `numpy` and `scipy` should be specified for compilation. These versions should be as low as possible to ensure broad compatibility across environments. However, specifying fixed versions introduces further complexity:

- Each `numpy` version is compatible only with specific Python versions. For example, `numpy 1.23.0` is compatible with Python versions 3.8 to 3.11 but not with 3.12.
- To address this, `pyproject.toml` must differentiate between Python versions and specify different `numpy` versions accordingly.

---

# PyPI Upload

Since the package is compiled on the user's system, only a source distribution can be uploaded to PyPI. To create such a distribution, use the following command:

```bash
python setup.py sdist
```

This will create a source distribution in the `dist/` directory. To upload the package to PyPI, you need the `twine` Python package. The upload command is as follows:

```bash
python -m twine upload --repository MHN dist/*
```

The `MHN` repository is defined in the `.pypirc` file as follows:

```ini
[MHN]
repository = https://upload.pypi.org/legacy/
username = __token__
password = <Insert API Token here>
```

More details can be found in the [Twine documentation](https://twine.readthedocs.io/en/stable/) and the [PyPI documentation on `.pypirc`](https://packaging.python.org/en/latest/specifications/pypirc/).

Additionally, the GitHub repository at https://github.com/StefanDevAccount/LearnMHN automates this process via a GitHub workflow. Administrators only need to create a new release to trigger the upload.  
If you want to use this functionality in your own fork of this repository, you have to add `PYPI_API_TOKEN` with your PyPI API token as value to the secrets of your GitHub repository.

**Never** upload a version without running all the tests located in the `test/` folder!

---

# setup.py

The `setup.py` script serves two primary purposes:

1. **Packaging for PyPI Upload:**  
   It packages the project for upload to PyPI using the command `python setup.py sdist` (see PyPI Upload).

2. **Installation Execution:**  
   During the installation of the package on the user's system, this script performs several tasks:

   - **System Analysis:**  
     It examines the user's system to determine the operating system and checks whether the CUDA compiler (`nvcc`) is installed.

   - **Package Definitions:**  
     The script contains several definitions critical for the package:
     - **Package Version:**  
       The `VERSION` in `setup.py` is passed to `setuptools` and stored in the package's metadata, allowing users to retrieve it via `__version__`.
     - **STATE_SIZE:**  
       This parameter defines the size of the binary state arrays used by the package. These arrays store the binary states of events (present or not present) as 
       `int32` arrays, where each event is represented by a single bit.  
       The `STATE_SIZE` determines the maximum number of events the package can handle, calculated as `32 * STATE_SIZE`.  
       By default, `STATE_SIZE` is set to `8`, allowing for a maximum of 256 events, which suffices for most use cases.  
       If larger models are required, this constant can be adjusted in `setup.py`, and the package reinstalled. Note that this requires recompilation of the entire Cython and CUDA code.

   - **CPU-Only Installation:**  
     The script defines an environment variable that can be used to enforce a CPU-only installation.

   - **Compilation:**  
     The script manages the compilation of the package:
     - **CUDA Compilation:**  
       If the CUDA compiler (`nvcc`) is available on the user's system, it compiles the CUDA code.
     - **Cython Compilation:**  
       The script converts Cython code to C code, which is then compiled by a C compiler on the user's system.  
       *Note:* A pre-installed C compiler is required for the package installation.

   - **Metadata File Creation:**  
     The script generates a `METADATA` file, which the package uses at runtime to access information such as the current package version.

---

# Other Top-Level Files

- **`pyproject.toml`:**  
  Contains only the requirements needed for the package installation itself (e.g., specific versions of Cython and NumPy).

- **`MANIFEST.in`:**  
  Specifies which non-Python files should be included in the final package. This includes files such as CUDA and Cython files, as well as the `METADATA` file.

- **`.readthedocs.yaml`:**  
  Provides instructions for automatically updating the documentation on [Read the Docs](https://readthedocs.org/) whenever new changes are pushed to the `main` branch.  
  *Note:* Both Read the Docs and a public GitHub repository must be properly configured for this functionality to work.
