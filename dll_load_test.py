import ctypes
import os

os.add_dll_directory(r"C:\Program Files\GTK3-Runtime Win64\bin")
ctypes.CDLL("libcairo-2.dll")
print("Cairo loaded successfully!")
print(f"paths:{os.environ['PATH'].split(os.pathsep)}")