# Issues and Suggestions

This document outlines potential issues, edge cases, and suggestions for the FCGMB codebase, specifically addressing the duplicate molecule issue raised and reviewing the project for organization, usability, and documentation.

## Critical Issues & Edge Cases

### 1. Duplicate Molecules in CSV Input
*   **Observation**: You mentioned resolving duplicate molecules using the median for bioactivity vs docking score. This logic is correctly implemented in `fcgmb/data.py` (`fetch_chembl_data`) for ChEMBL retrievals.
*   **The Issue**: This deduplication logic is **bypassed** when a user provides a local CSV file via the `ligand_csv_path` configuration in `run_workflow.py`.
    *   In `fcgmb/workflow.py`, the CSV is loaded and only column renaming is performed.
    *   If the input CSV contains duplicates (e.g., multiple bioactivity measurements for the same SMILES), `run_workflow.py` will dock the unique SMILES but then join the results back to the original DataFrame with duplicates.
    *   **Result**: The final output and plots will contain multiple points for the same molecule (same docking score, different or same bioactivity), which might be unintended if you want a 1:1 molecule-to-score relationship.
*   **Suggestion**: Apply the same aggregation logic (group by Canonical SMILES -> Median Bioactivity) to the loaded CSV data in `fcgmb/workflow.py`.

### 2. Standardization Bypass for CSV Input
*   **Observation**: `fetch_chembl_data` applies rigorous standardization (stripping salts, neutralizing, canonicalizing) using `fcgmb.data.standardize_smiles`.
*   **The Issue**: When using `ligand_csv_path`, this standardization step is skipped.
*   **Edge Case**: If the user's CSV contains non-canonical SMILES or salts (e.g., `[Na+].[Cl-].CCO`), they might not match the fragment constraints or could be treated as different molecules from their canonical forms.
*   **Suggestion**: call `standardize_smiles` on the `canonical_smiles` column of the loaded CSV DataFrame.

## Codebase Review

### Organization
*   **Pros**: The project is well-structured. Core logic is cleanly separated into `data`, `docking`, `receptor`, and `workflow` modules within the `fcgmb` package. The `fcgmb/pipeline` directory correctly isolates benchmark generation scripts.
*   **Cons**: `fcgmb/docking.py` (specifically the `AutoDockGPUOracle` class) is becoming large (~500 lines). It handles ligand preparation (Meeko/Scrub), ADGPU execution, and result parsing/filtering.
    *   **Suggestion**: Consider extracting the `_prepare_single_ligand` logic into a dedicated `LigandPreparer` class and result parsing into a `DockingResultParser` class to improve maintainability.

### Usability & Modifiability
*   **Ease of Start**: The package is easy to install and the `run_workflow.py` script serves as a clear entry point. The CLI arguments are intuitive.
*   **Modifiability**: The code is modular. A user can easily modify `fcgmb/docking.py` to change docking parameters without affecting data retrieval.
*   **Integration**: The `FCGMBOracle` class provides a fantastic high-level API (`oracle.score_batch`), making it very easy for generative models to interface with the benchmark.
*   **Suggestion**: Currently, `FCGMBOracle` seems to rely on internal benchmarks or a specific directory structure. Allowing it to accept a generic dictionary configuration or path to a custom YAML file would make it more flexible for users testing new systems not yet bundled.

### Documentation
*   **Pros**: `README.md` covers installation and basic usage well.
*   **Cons**: There is limited documentation on how to **create new benchmarks**. The `fcgmb/pipeline` folder contains scripts for this (`generate_benchmark_configs.py`), but a user might not know how to use them to generate the necessary grids and config files for a new protein target.
*   **Suggestion**: Add a "Creating Custom Benchmarks" section to the documentation, explaining the workflow from PDB ID -> Grid Generation -> Config creation using the scripts in `fcgmb/pipeline`.

### Benchmarking
*   **Analysis**: The variance analysis tools (`run_variance.py`) are a great addition for scientific rigor.
*   **Batching**: While `run_variance.py` runs multiple iterations, there isn't a direct "run all benchmarks" script for a single pass (though `run_variance.py` could be adapted). A simple `run_batch.py` that iterates over all YAMLs in a folder would be helpful for large-scale evaluations.

## Summary of Recommendations

1.  **Fix CSV Duplicates**: Update `fcgmb/workflow.py` to standardize and deduplicate CSV inputs (using median aggregation) to match ChEMBL retrieval behavior.
2.  **Receptor Prep**: Integrate **PDBFixer** and **PDB2PQR** into `fcgmb/receptor.py` to ensure high-quality receptor inputs.
3.  **Refactor Docking**: Break down `AutoDockGPUOracle` if it grows further.
4.  **Document Pipeline**: Explain how to use `fcgmb/pipeline` to generate new benchmarks.
5.  **Enhance Oracle**: Allow `FCGMBOracle` to load custom configs easily.

## Recent Updates & Proposals

### 1. Simplified Scratch Directory Structure
*   **Issue**: The current scratch directory (`.fcgmb` or user-defined) creates an excessively deep and redundant hierarchy, e.g., `grids/<pdb_id>/grid/` and `benchmarks/<name>/results/results/`.
*   **Proposed Structure**: Flatten the directory layout to improve navigation and reduce path length issues.
    ```text
    .fcgmb/
    ├── grids/
    │   └── <pdb_id>/              # Store map files directly here (remove 'grid' subfolder)
    │       ├── <pdb_id>.maps.fld
    │       └── ...
    ├── data/                      # Renamed from 'ligand_data' for brevity
    │   └── <cached_chembl.csv>
    └── runs/                      # Renamed from 'benchmarks'
        └── <benchmark_name>/
            └── results/           # Store DLGs/results directly here (remove second 'results' folder)
                └── <dlgs>
    ```
*   **Implementation Notes**:
    *   Update `FCGMBOracle.__init__` paths.
    *   Update `AutoDockGPUOracle` to accept a direct output path rather than appending `/results` automatically, or ensure `FCGMBOracle` passes the parent.
    *   Update `ReceptorPreparer` to output grid files directly to the PDB folder.

### 2. Receptor Preparation Update
*   **Status**: The `ReceptorPreparer` class now successfully integrates **Reduce2** (`mmtbx.reduce2`) for adding hydrogens, replacing previous methods.
*   **Next Steps**: Continue to monitor the robustness of Reduce2 on diverse PDBs. The suggestion to add PDBFixer remains relevant for fixing missing atoms/residues prior to hydrogenation if Reduce2 encounters issues with poor quality structures.

### 3. README & Installation Improvements
*   **Package Management**: Explicitly document installation using `uv` for fast, reliable dependency management.
    ```bash
    # Example
    uv pip install -e .
    ```
*   **Binary Dependencies**: The README must clearly list external binaries required for the workflow, as they are not installed via Python:
    *   `adgpu` (AutoDock-GPU)
    *   `autogrid4` (AutoGrid4)
    *   `mmtbx.reduce2` (Phenix/Reduce)
*   **Environment Variables**: Explain how to set PATH or pass executable paths to `FCGMBOracle` if binaries are in non-standard locations.

### 4. Feature Request: AutoDock Vina Support
*   **Goal**: Allow users to switch between AutoDock-GPU and AutoDock Vina.
*   **Implementation Plan**:
    1.  **Create `VinaOracle`**: Subclass `DockingOracle` in `fcgmb/docking.py`.
        *   Accept `vina_executable` path.
        *   In `dock_batch`, use `subprocess` to call Vina:
            ```bash
            vina --receptor <rec_pdbqt> --ligand <lig_pdbqt> --center_x <x> ... --size_x <x> ... --cpu <n>
            ```
        *   **Note**: Vina does not use `.maps.fld` (AutoGrid maps) by default but needs the center/size configuration. This metadata is currently generated during the grid prep. `FCGMBOracle` or `ReceptorPreparer` should expose the grid box center/dimensions (parsed from `.gpf` or `.maps.fld`) to the `VinaOracle`.
    2.  **Update `FCGMBOracle`**:
        *   Add `docking_backend` argument (default: "adgpu").
        *   If `docking_backend="vina"`, instantiate `VinaOracle` instead of `AutoDockGPUOracle`.
        *   Ensure `_ensure_components` passes the necessary grid parameters (box center/size) to the Vina oracle, as Vina computes grids internally.
    3.  **User Experience**:
        *   Config YAML update: Add `docking_backend: vina`.
        *   Execution: `fcgmb-run ... --backend vina` (if CLI exposed) or just via config.
