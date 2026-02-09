# VIASM Codes for Summer School on ANM 2025

Hands-on codes for [VIASM 2025 summer school on advanced numerical methods for deterministic and stochastic differential equations](https://viasm.edu.vn/en/hdkh/summer-school-on-anm2025).

The lectures that introduce and motivate these codes are available on Google Drive: https://tinyurl.com/viasm-anm2025.

## Setup

These codes require a modern Python installation, a few standard Python modules, and the `git` command for interacting with GitHub.  The following step-by-step instructions guide you in installing Anaconda, downloading code from GitHub, and running the code in Spyder.  These steps are suitable for both Windows and MacOS users.

### Install Anaconda

**For Windows & macOS**

1. Go to the Anaconda website:

   https://www.anaconda.com/download

2. Download the Installer:

   * Click **Download**

   * Choose the version for your operating system (Windows or MacOS).

   * Download the **64-bit Graphical Installer**.

3. **Install Anaconda:**

   * **Windows**: Double-click the `.exe` file and follow the setup instructions.

   * **Mac**: Double-click the `.pkg` file and follow the prompts.

   *Tip: Leave all settings at their defaults unless you have specific needs. On
    Windows, you may want to check "Add Anaconda to my PATH environment variable"
    (optional but  useful).*

### Launch Anaconda Navigator

Open **Anaconda Navigator** from your Start Menu (Windows) or Applications folder
(MacOS).

### Download Code from GitHub

#### Option 1: Using Git (Recommended)

1. **Install Git** (if you don't have this already):

   Download from https://git-scm.com/downloads and install using default settings.

2. **Open Anaconda Prompt (Windows) or Terminal (Mac).**

3. **Navigate to the folder where you want to save the code:**

   ```bash
   cd path/to/your/folder
   ```

   where you replace `path/to/your/folder` with the path to your desired folder.

4. **Clone the repository:**

   ```bash
   git clone https://github.com/drreynolds/viasm.git
   ```

#### Option 2: Download ZIP (No Git Required)

1. Go to the GitHub repository page in your web browser.

2. Click the green **Code button**, then select **Download ZIP**.

3. Extract the ZIP file to your preferred location.

### Installing Python dependencies

To install the Python packages that are used by the codes in this repository, use **Anaconda Prompt/Terminal** from the folder containing the downloaded/cloned code:

```bash
pip install -r python_requirements.txt
```

### Running the codes

#### Option 1: run directly in Anaconda Prompt/Terminal

Run the desired Python script directly at the command-line:

```bash
python scriptname.py
```

where `scriptname.py` is the name of the script you wish to run.

#### Option 2: run in Spyder

1. Launch Spyder:

   * In Anaconda Navigator, locate **Spyder**.

   * If not already installed, click **Install.**

   * After installation, click **Launch** to open Spyder.

2. Open and run the code in Spyder

   1. In Spyder, go to **File > Open**.

   2. Navigate to the folder containing the downloaded/cloned code.

   3. Select the `.py` file you want to run and open it.

   4. Press the green **Run** button (Play icon) in Spyder to execute your script.

## File layout

Files that begin with capital letters contain Python classes and user-callable functions.  Files that begin with lower-case letters contain scripts that run various demos.  These are grouped as follows:

### Background

Basic Python demos:

* `numpy_demo.py` -- simple script showing Numpy usage.

* `plotting_demo.py` -- simple script showing how to generate plots.

### Fixed-step Runge--Kutta methods and implicit solvers

* `ForwardEuler.py` -- simple baseline explicit IVP time-stepper class.

* `ERK.py` -- explicit Runge--Kutta IVP time-stepper class.

* `driver_explicit_fixed.py` -- script to test explicit fixed-step methods.

* `driver_explicit_stability.py` -- script to demonstrate stability limitations of explicit methods.

* `ImplicitSolver.py` -- reusable nonlinear solver class for implicit IVP methods.

* `BackwardEuler.py` -- simple baseline implicit IVP time-stepper class.

* `DIRK.py` -- diagonally-implicit Runge--Kutta IVP time-stepper class.

* `driver_implicit_fixed.py` -- script to test implicit fixed-step methods.

### Adaptive-step Runge--Kutta methods

* `AdaptERK.py` -- explicit Runge--Kutta adaptive IVP solver class.

* `driver_explicit_adaptive.py` -- script to test explicit adaptive-step methods.

* `AdaptDIRK.py` -- implicit Runge--Kutta adaptive IVP solver class.

* `driver_implicit_adaptive.py` -- script to test implicit adaptive-step methods.

* `driver_adaptive_timescale.py` -- script to demonstrate how adaptive solvers track dynamical time scales.

* `driver_adaptive_stability.py` -- script to demonstrate how adaptive solvers can assess stiffness.

### Multirate methods

* `LTSubcycling.py` -- simple fixed-step Lie--Trotter subcycling IVP time-stepper class.

* `SMSubcycling.py` -- simple fixed-step Strang--Marchuk subcycling IVP time-stepper class.

* `MRI.py` -- higher-order fixed-step multirate infinitesimal (MRI) IVP time-stepper class.

* `driver_multirate.py` -- script to demonstrate accuracy differences for various multirate methods.

### Auxiliary utilities

* `RK_stability.py` -- function to plot linear stability regions for Runge--Kutta methods.

## Authors

[Daniel R. Reynolds](https://drreynolds.github.io/)  
[Mathematics & Statistics @ UMBC](https://mathstat.umbc.edu)

[Van Hoang Nguyen](https://www.depts.ttu.edu/math/facultystaff)  
[Mathematics & Statistics @ TTU](https://www.depts.ttu.edu/math)
