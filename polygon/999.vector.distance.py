#!/usr/bin/env python3

import torch
import matplotlib.pyplot as plt

# Define vectors
A = torch.tensor([2.0, 4.0])
B = torch.tensor([4.0, 8.0])  # Same line
C = torch.tensor([5.0, 0.0])  # Different direction

# Distances
dist_AB = torch.dist(A, B).item()
dist_AC = torch.dist(A, C).item()

# Cosine similarities
cos_AB = torch.nn.functional.cosine_similarity(A.unsqueeze(0), B.unsqueeze(0)).item()
cos_AC = torch.nn.functional.cosine_similarity(A.unsqueeze(0), C.unsqueeze(0)).item()

# Print metrics
print(f"Distance A-B: {dist_AB:.2f}")
print(f"Distance A-C: {dist_AC:.2f}")
print(f"Cosine similarity A-B: {cos_AB:.2f}")
print(f"Cosine similarity A-C: {cos_AC:.2f}")

# Plot
plt.figure(figsize=(8, 8))
plt.quiver(0, 0, A[0], A[1], angles='xy', scale_units='xy', scale=1, color='r', label='A (2, 4)')
plt.quiver(0, 0, B[0], B[1], angles='xy', scale_units='xy', scale=1, color='g', label='B (4, 8)')
plt.quiver(0, 0, C[0], C[1], angles='xy', scale_units='xy', scale=1, color='b', label='C (5, 0)')

plt.xlim(-1, 9)
plt.ylim(-1, 9)
plt.grid()
plt.legend()
plt.title("Vectors A, B, C and their relations")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()
