# Project plan
- Implement linear advection in fenics (project 1 from ANM) (order 2 conv.)
- Implement burger’s equation, use artificial diffusion (RV)
- Implement KPP equation, use artificial diffusion (RV) (project 2)
- Do burger’s equation again, but with smoothness indicator
- Do KPP equation again, but with smoothness indicator
- Implement compressible euler equations (vector) with artificial diffusion
- Implement compressible euler equations with smoothness indicator
- Calculate CFL lines (rankgine hugoniat)

For all of the above, convergence rate is to be reported for mesh sizes 1/4, 1/8, 1/16 and 1/32. FEniCSx (the new version of FEniCS) is used.

# Progression map
## w. 45
We set up a FEniCSx environment locally on our computers. We have also read the FEniCSx documentation available [here](https://jsdokken.com/dolfinx-tutorial/index.html). Then we implemented linear advection and calculated the convergence rate for mesh sizes 1/4, 1/8, 1/16. 1/32 was not included since FEniCSx couldn't resolve it in the convergence loop for unkown reasons. We have briefly started investigating how to implement artificial viscosity in FEniCSx.