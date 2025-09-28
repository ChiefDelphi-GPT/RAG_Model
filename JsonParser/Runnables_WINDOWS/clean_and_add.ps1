$ErrorActionPreference = "Stop"

if ($args[0] -eq "--fresh") {
    .\clean_inputs.ps1 --fresh
}

.\clean_inputs.ps1
.\add_vectors.ps1