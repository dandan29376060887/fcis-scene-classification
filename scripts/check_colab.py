"""Explain how to verify a Google Colab runtime safely.

Colab runtimes are managed by Google and do not provide a stable SSH
endpoint. Run this notebook's GPU check inside Colab instead of trying
to connect to a runtime with hard-coded credentials from a local PC.
"""

print("Open colab_train.ipynb in Google Colab and run its first cell.")
print("That cell verifies CUDA and prints the assigned GPU model.")
