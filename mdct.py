"""
===============================================================================
Title:        mdct.py
Description:  1-D MDCT using the DCT-IV
Author:       Sushmita Raj
Date:         03.12.2024
Git:          https://github.com/fortytwobits/MDCT.git
===============================================================================
Dependencies:
- Python 3.12.7+
- Libraries: NumPy, SciPy, Matplotlib

License:
This software is licensed under the MIT License.
You may use, copy, modify, and distribute this software under the terms of the
MIT License. See the LICENSE file in the repository or the full license text 
below.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.

Usage:
Run the program to run the included test script
    python mdct.py
Use the functions mdct and imdct to calculate the mdct and imdct of a 1D signal

Description:
Input: 1-D signal of size NxM
N=Block size, M=number of non-overlapped blocks

MDCT:
1. Padding: Pad with N zeroes at the beginning and end of the signal
2. Overlap: Blocks of size N are rearranged into overlapping blocks of size 2N
overlap between consecutive blocks = N samples
3. Window: Multiply with a window that satisfies the Princen-Bradley condition
ex. rectangular, sine, Kaiser-Bessel Derived, Vorbis
4. Transform: Split each block of length 2N into 4 sub-blocks 
each of size N/2 (a, b, c, d) 
5. The MDCT of each block is equivalent to a DCT-IV of the N samples:
(−cR−d, a−bR), where R denotes reversal of the sequence

IMDCT:
1. IDCT gives us back the blocks (−cR−d, a−bR) of length N
This can be used to calculate the IMDCT blocks of length 2N
2. IMDCT (MDCT (a, b, c, d)) = (A, B, C, D) where
A = a-bR
B = b−aR = −(a−bR)R
C = c+dR = -(-cR-d)R
D = d+cR = -(-cR-d)
3. Window: Multiply again with the chosen window
4. Reconstruction: Add the overlapping parts of adjacent blocks to get back 
the original sequence

Note:
Not the most efficient implementation as the purpose of this code is learning
about and understanding the concepts

References and Resources:
https://en.wikipedia.org/wiki/Modified_discrete_cosine_transform
https://web.archive.org/web/20230227100852/https://www.appletonaudio.com/
blog/2013/understanding-the-modified-discrete-cosine-transform-mdct/
https://github.com/TUIlmenauAMS
https://github.com/TUIlmenauAMS/MRSP_Tutorials/blob/master/MRSP_mdct.ipynb
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sfft

def mdct(signal, N=8, window=None):
    """
    Perform the Modified Discrete Cosine Transform (MDCT) on a signal.

    Parameters:
    - signal (numpy array): Input 1D signal.
    - N (int): Block size for the MDCT (default is 8). Must be even.
    - window (numpy array or None): Optional window function of size 2*N.
      If None, no windowing is applied.

    Returns:
    - X (numpy array): MDCT coefficients as a 2D array, where each row represents 
      the MDCT of a block of the input signal.
    """
    # Calculate the number of blocks (M) and half the block size (N/2)
    M = int(signal.shape[0] / N)  # Number of non-overlapping blocks
    halfN = int(N / 2)

    # Pad the signal with zeros to ensure full block coverage
    # Padding length is N on both sides
    x_padded = np.pad(signal, pad_width=N, mode='constant', constant_values=0)

    # Split the padded signal into M+1 overlapping blocks of size 2*N
    blocks = np.array([x_padded[i:i + 2 * N] 
                       for i in range(0, len(x_padded) - 2 * N + 1, N)])

    # Apply the provided window function to each block, if specified
    # If no window is provided, the blocks remain unchanged
    winblocks = (window * blocks) if window is not None else blocks

    # Initialize an array for storing the overlapped and transformed blocks
    lapblocks = np.zeros((M + 1, N))

    # Overlap-Add Process: Combine overlapping portions of blocks
    for i in range(M + 1):
        # Split the current block into four equal parts
        a = winblocks[i, 0:halfN]         # First quarter
        b = winblocks[i, halfN:N]        # Second quarter
        c = winblocks[i, N:3*halfN]      # Third quarter
        d = winblocks[i, 3*halfN:2*N]    # Fourth quarter

        # Reverse the second and third quarters
        br = b[::-1]  # Reverse of `b`
        cr = c[::-1]  # Reverse of `c`

        # Debug print to observe the segments if you wish
        # print("a br cr d", a, br, cr, d)

        # Combine the segments using the MDCT overlap-add equations
        A = -cr - d  # Compute the left-half coefficients
        B = a - br   # Compute the right-half coefficients

        # Concatenate A and B to form the current overlapped block
        lapblocks[i] = np.concatenate((A, B))

    # Apply the Type-4 Discrete Cosine Transform (DCT-IV) to each block
    # The result is a 2D array where each row is the MDCT of a block
    X = np.zeros((M + 1, N))
    for i in range(M + 1):
        X[i] = sfft.dct(lapblocks[i], type=4, norm="ortho")

    X_f = X.flatten()

    # Return the MDCT coefficients
    return X_f

def imdct(subbands, N=8, window=None):
    """
    Perform the Inverse Modified Discrete Cosine Transform (IMDCT) on a set of 
    MDCT coefficients.

    Parameters:
    - subbands (numpy array): A 2D array of MDCT coefficients, where each row 
        corresponds to the MDCT of a signal block.
    - N (int): Block size used during the MDCT (default is 8). 
        Must be an even integer.
    - window (numpy array or None): An optional window function of length 2*N. 
        If None, no(/rectangular) windowing is applied during reconstruction

    Returns:
    - x_recon (numpy array): The reconstructed 1D signal from the IMDCT process
    """
    # Determine the number of overlapping blocks (M+1) from the input subbands
    M = int(subbands.shape[0]/N) - 1  # Non-overlapping blocks, minus 1 due to overlap
    halfN = int(N / 2)         # Half the block size, used in segmenting blocks

    subbands = np.reshape(subbands, (M+1,N))
    # Initialize an array to store the output of the Inverse DCT-IV (IDCT-IV)
    # IDCT-IV is applied block-by-block to the MDCT coefficients (subbands)
    Xd = np.zeros((M + 1, N))
    for i in range(M + 1):
        # Apply IDCT-IV, 
        # IDCT-IV is mathematically equivalent to DCT-IV (self-inverse)
        Xd[i] = sfft.dct(subbands[i], type=4, norm="ortho")

    # Initialize arrays for intermediate reconstruction steps
    z = np.zeros((M + 1, 2 * N))  # overlapped blocks in the time domain

    # Reconstruct the overlapping blocks using the IMDCT algorithm
    for i in range(M + 1):
        # Split the IDCT-IV output into two halves
        A = Xd[i, halfN:]                # Second half
        B = -np.flip(Xd[i, halfN:])      # Flipped and negated second half
        C = -np.flip(Xd[i, :halfN])      # Flipped and negated first half
        D = -Xd[i, :halfN]               # First half, negated

        # Concatenate the four parts to form the reconstructed overlapped block
        z[i, :] = np.concatenate((A, B, C, D))

    # Apply the synthesis window if provided, otherwise keep blocks unchanged
    winz = (window * z) if window is not None else z

    # Initialize the final reconstructed signal blocks
    x_recon = np.zeros((M, N))  # the M original non-overlapping blocks

    # Perform overlap-add synthesis to reconstruct each block
    for i in range(M):
        # Combine the overlapping portions of consecutive blocks
        x_recon[i, :] = winz[i, N:] + winz[i + 1, :N]

    # Flatten the 2D array of reconstructed blocks into a single 1D signal
    x_recon = x_recon.flatten()

    return x_recon

#Testing:
if __name__ == '__main__':
    N = 8 # block size
    M = 64 # number of blocks
    x = np.arange(M*N) # ramp input signal
    window = np.sin((np.pi/(2*N)*((np.arange(2*N))+0.5)))

    # MDCT
    X = mdct(x, N, window)
    # IMDCT
    x_recon = imdct(X, N, window)

    # Print the reconstructed sequence if you wish
    # print("x reconstructed", x_recon)

    # Reconstruction error
    # Calculate Reconstruction Error
    error_signal = x - x_recon
    mse = np.mean(error_signal**2)
    rmse = np.sqrt(mse)
    relative_error = np.linalg.norm(error_signal) / np.linalg.norm(x)
    print(f"Mean Squared Error: {mse}")
    print(f"Root Mean Squared Error: {rmse}")
    print(f"Relative Error: {relative_error}")

    # Plot the original and reconstructed signals against each other
    plt.figure(figsize=(10, 6))
    plt.plot(x, label='Original Signal', color='blue', linewidth=2)         # Original signal
    plt.plot(x_recon, label='Reconstructed Signal', color='orange', linestyle='--')  # Reconstructed signal
    plt.title("Original vs Reconstructed Signal")
    plt.xlabel("Sample Index")
    plt.ylabel("Amplitude")
    plt.legend()  # Show the legend
    plt.grid()    # Add grid lines
    plt.show()
