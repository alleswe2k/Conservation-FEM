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

## W.49
We continued working on the report, including our progress in the results as well as writing in other sections as well. Apart from the writing, we started implementing a solver for the Euler system of equations in FEniCSx. We also had our mock presentation. 

## W.48
We have been working on our presentation, and figuring out the structure of the report, what to include and what not. Moreover, we have been testing different normalization methods, debugging the smoothness indicator code as well as plotting everything that is necessary.

## W. 47 
We have gone through the code and verified each part, the code now yields more accurate figures regarding the exact burgers and KPP. We have also tested the BDF2 method of calculating RH for burgers and KPP as well as a different approach of normalizing the residual. We went through our Github repo and restructured the folder for an easier time navigating it. 


## w. 46
We've implemented residual viscosity, both cell based and nodal based. This was tested on the
linear advection PDE. We've also implemented the KPP and Burger PDE, and stabilized both with nodal
based RV. We've also implemented the smoothness indicator and tested convergence on the linear advection
problem. Residual plots have been generated for RV and smoothness indicator.

## w. 45
We set up a FEniCSx environment locally on our computers. We have also read the FEniCSx documentation available [here](https://jsdokken.com/dolfinx-tutorial/index.html). Then we implemented linear advection and calculated the convergence rate for mesh sizes 1/4, 1/8, 1/16. 1/32 was not included since FEniCSx couldn't resolve it in the convergence loop for unkown reasons. We have briefly started investigating how to implement artificial viscosity in FEniCSx.