"""
===============================================================================
Title:        mdct2.py
Description:  2-D MDCT using 1-D MDCT
Author:       Sushmita Raj
Date:         04.12.2024
Git:          https://github.com/fortytwobits/MDCT.git
===============================================================================
Dependencies:
- Python 3.12.7+
- Libraries: NumPy, Matplotlib
- mdct.py (contains the 1-D MDCT implementation)

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
    python mdct2.py
Use the functions mdct2 and imdct2 on 2-D signals

Description:
x = 2-D signal of size (N*M, N*M)
N = block size
M = number of blocks
window = window that satisfies the Princen-Bradley condition
2-D MDCT: Apply the 1-D MDCT first along the rows and then along the column
2-D IMDCT: Apply the 1-D IMDCT first along the columns and then along the rows
===============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from mdct import mdct, imdct

def mdct1(x, N, window=None):
    M = int(x.shape[1]//N) # Number of non-overlapping blocks
    nBlocks = x.shape[0]
    outlen = N*(M+1)

    X = np.zeros((nBlocks,outlen))

    for ri in range(nBlocks):
        X[ri] = np.reshape(mdct(x[ri], N, window), outlen)

    normFactor = np.sqrt(1/2) if window==None else 1
    
    return normFactor*X

def imdct1(X, N, window=None):
    M = int(X.shape[1]//N) - 1 # Number of non-overlapping blocks
    nBlocks = X.shape[0]
    outlen = N*M
    
    x = np.zeros((nBlocks,outlen))

    for ri in range(nBlocks):
        x[ri] = np.reshape(imdct(X[ri], N, window), outlen)

    normFactor = np.sqrt(1/2) if window==None else 1
    
    return normFactor*x

def mdct2(x, N, window=None, showAll=False):
    xr_mdct = mdct1(x, N)
    x_mdct = mdct1(xr_mdct.T, N).T
    
    if showAll:
        # Display original signal
        print("x", x.shape, x)
        plt.imshow(x, cmap='viridis', interpolation='nearest')
        plt.colorbar()  # Show color scale
        plt.title('Original')
        plt.show()

        # Display MDCT along rows
        plt.imshow(xr_mdct, cmap='viridis', interpolation='nearest')
        plt.colorbar()  # Show color scale
        plt.title('MDCT along rows')
        plt.show()

        # Display MDCT along rows and columns
        print("X MDCT",x_mdct.shape, x_mdct)
        plt.imshow(x_mdct, cmap='viridis', interpolation='nearest')
        plt.colorbar()  # Show color scale
        plt.title('MDCT along rows and columns')
        plt.show()

    return x_mdct

def imdct2(X, N, window=None, showAll=False):
    xc_imdct = imdct1(X.T, N).T
    x_imdct = imdct1(xc_imdct, N)

    if showAll:
        # Display original signal
        plt.imshow(X, cmap='viridis', interpolation='nearest')
        plt.colorbar()  # Show color scale
        plt.title('Original Subbands')
        plt.show()

        # IMDCT along columns
        plt.imshow(xc_imdct, cmap='viridis', interpolation='nearest')
        plt.colorbar()  # Show color scale
        plt.title('IMDCT along columns')
        plt.show()

        # IMDCT along columns and rows
        print("X IMDCT",x_imdct.shape, x_imdct)
        plt.imshow(x_imdct, cmap='viridis', interpolation='nearest')
        plt.colorbar()  # Show color scale
        plt.title('IMDCT along columns and rows')
        plt.show()

    return x_imdct
    
#Testing:
if __name__ == '__main__':
    N = 4
    M = 2
    # Input signal
    x = np.arange(1, (N*M)**2 + 1).reshape((N*M), (N*M))
    win = np.sin((np.pi/(2*N)*((np.arange(2*N))+0.5)))
    X = mdct2(x, N, win, True)
    x_recon = imdct2(X, N, win, True)
    print(x_recon)

    # Plot
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 3, 1)
    plt.title("Original Signal")
    plt.imshow(x, cmap='viridis', aspect='auto')
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Subbands")
    plt.imshow(X, cmap='viridis', aspect='auto')
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Reconstructed Signal")
    plt.imshow(x_recon, cmap='viridis', aspect='auto')
    plt.colorbar()

    plt.tight_layout()
    plt.show()

    # Mean Absolute Error (MAE)
    mae = np.mean(np.abs(x - x_recon))
    print(f"Mean Absolute Error (MAE): {mae}")

    # Mean Squared Error (MSE)
    mse = np.mean((x - x_recon) ** 2)
    print(f"Mean Squared Error (MSE): {mse}")

    # Root Mean Squared Error (RMSE)
    rmse = np.sqrt(mse)
    print(f"Root Mean Squared Error (RMSE): {rmse}")

    # Peak Signal-to-Noise Ratio (PSNR)
    max_val = np.max(x)  # Assuming the original signal defines the max possible value
    psnr = 20 * np.log10(max_val / rmse) if rmse > 0 else float('inf')
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr} dB")
