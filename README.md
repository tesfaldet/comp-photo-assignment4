# comp-photo-assignment4

#### Task 2: Flash/No-Flash Photography
This task follows the same implementation strategies as that of the Flash/No-Flash paper. No deviations whatsoever.

#### Task 3: Multi-flash Camera (depth-edge detection)
For capturing the depth edges at each flash orientation, a 1D convolution with an appropriate 1D kernel is used to detect "negative transitions", i.e. regions that change from a bright intensity to a dark intensity (shadow). After all
depth edges at each orientation are found, they are summed to produce the final depth edge output. The result is cleaned up and applied to the original image to emphasize its depth edges.
