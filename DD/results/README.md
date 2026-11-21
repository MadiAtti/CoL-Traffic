This directory contain all the raw data and scipts to get the results as written in the paper.
The directory contains two use cases - Drug Discovery (DD), and Recommender System (RS).

Each use case contain:
- The plots for the small and large data
- The plots for their corresponding SD
For these two, I added also the CSVs before and after the average.
- The data saturation analysis (for now it missing in the RS use case, since it is still running and will send later)
- Appendix A experiments with different ratios
- The transformed SD function with the optimal weights and RMSE values for Tables 2 & 3
- The difference between the averaged plots and the SD plots applied the transfomed SD function

Each experiment have a python scirpt that upon running will generate the current directory results.