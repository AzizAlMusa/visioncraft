try:
    import pyfftw
    print("pyFFTW is installed. Version:", pyfftw.__version__)
except ImportError:
    print("pyFFTW is NOT installed.")

try:
    import cupy
    print("CuPy is installed. Version:", cupy.__version__)
except ImportError:
    print("CuPy is NOT installed.")
