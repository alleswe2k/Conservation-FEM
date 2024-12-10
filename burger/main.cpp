#include <fstream>
#include <dolfin.h>
#include "Burger.h"

using namespace dolfin;
const double pi = DOLFIN_PI;
const double xmin = 0.0;
const double xmax = 1.0;
const double ymin = 0.0;
const double ymax = 1.0;

// 
//------------------------------------------------------------------------
class Indata : public Expression
{
public:

  // Evaluate pressure at inflow
  void eval(Array<double>& values, const Array<double>& x) const
  {
    // // // values[0] = sin(0.5 * x[0]) * sin(0.5 * x[1]);
    // values[0] = std::exp(- (pow(x[0]-0.5, 2) + pow(x[1]-0.5, 2))/0.0125);
    // // //values[0] = 1.-2.*dolfin::rand();

    //-------------------------------------------------------------------
    double r0 = 0.2;
    double a = 0.5;
    double R = sqrt(pow(x[0] - 0.5, 2) + pow(x[1] - 0.5, 2));
    if (R <= r0)
      values[0] = 1.;
    else
     values[0] = 0.;
  }

};

// 
//------------------------------------------------------------------------
class ExactSolution : public Expression
{
public:

  // Constructor
  ExactSolution() : t(0) {}

  // Evaluate pressure at inflow
  void eval(Array<double>& values, const Array<double>& x) const
  {
    values[0] = std::exp(-0.25 * t) * sin(0.5 * x[0]) * sin(0.5 * x[1]);
  }

  // Current time
  double t;
};

// Sub domain for Dirichlet boundary condition
class DirichletBoundary : public SubDomain
{
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return on_boundary;
  }
};

///
class Burger_Exact_sol: public Expression
{
public:
  Burger_Exact_sol(double& t) :
      t(t)
  {
  }
  
  void eval(Array<double>& values, const Array<double>& x) const
  {
    double const eps_ = 1e-8;
    
    if (x[0] < 0.5 - 3.0 / 5.0 * t)
    {
      if (x[1] > 1. / 2. + 3.0 / 20.0 * t - eps_)
      {
        values[0] = -0.2;
      }
      else
      {
        values[0] = 0.5;
      }
    }

    else if (1. / 2. - 3.0 / 5.0 * t <= x[0] && x[0] < 1.0 / 2.0 - t / 4.0)
    {
      if (x[1] > -8. / 7. * x[0] + 15. / 14. - 15. / 28. * t - eps_)
      {
        values[0] = -1.;
      }
      else
      {
        values[0] = 0.5;
      }
    }

    else if (0.5 - 1. / 4. * t <= x[0] && x[0] < 1.0 / 2.0 + t / 2.0)
    {
      if (x[1] > 1. / 6. * x[0] + 5. / 12. - 5. / 24. * t - eps_)
      {
        values[0] = -1.;
      }
      else
      {
        values[0] = 0.5;
      }
    }

    else if (0.5 + 1. / 2. * t <= x[0] && x[0] < 1.0 / 2.0 + 4. / 5. * t)
    {
      if (x[1] > x[0] - 5. / (18. * t) * pow(x[0] + t - 1. / 2., 2) - eps_)
      {
        values[0] = -1.;
      }
      else
      {
        values[0] = (2. * x[0] - 1.) / (2. * t);
      }
    }

    else if (1. / 2. + 4. / 5. * t <= x[0])
    {
      if (x[1] > 1. / 2. - t / 10. - eps_)
      {
        values[0] = -1.;
      }
      else
      {
        values[0] = 0.8;
      }
    }
  }
  
private:
  double& t;
};

///
//------------------------------------------------------------------------
//Sub domain for Periodic boundary condition
class PeriodicBoundaryXY1: public SubDomain
{
  double const eps = 1.0e-8;

  // Left boundary is "target domain" G
  bool inside(const Array<double>& x, bool on_boundary) const
  {
    return
      (
       (x[0] < xmin + eps) || (x[1] < ymin + eps)
       )
      &&
      !(
	((x[0] > xmax - eps) && (x[1] < ymin + eps))
	||
	((x[0] < xmin + eps) && (x[1] > ymax - eps))
	)
      &&
      on_boundary;
  }

  // Map right boundary (H) to left boundary (G)
  void map(const Array<double>& x, Array<double>& y) const
  {

    if ((x[0] > xmax - eps) && (x[1] > ymax - eps))
      {
	y[0] = x[0] - (xmax - xmin);
	y[1] = x[1] - (ymax - ymin);
      }
    else if (x[0] > xmax - eps)
      {
	y[0] = x[0] - (xmax - xmin);
	y[1] = x[1];
      }
    else if (x[1] > ymax - eps)
      {
	y[0] = x[0];
	y[1] = x[1] - (ymax - ymin);
      }
    else
      {
	y[0] = -1000.0;
	y[1] = -1000.0;
      }
  }
};


//-----------------------------------------------------------------------------
void compute_alphaij(Matrix const & A, Function const & Fu, Function & Falpha)
{
  std::vector<double> Fu_arr(Fu.vector()->local_size());
  Fu.vector()->get_local(Fu_arr);

  std::vector<double> Falpha_arr(Falpha.vector()->local_size());

  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates0;

  const GenericDofMap& dofmap_u = *Falpha.function_space()->dofmap();
  dolfin::la_index dofsize_u = dofmap_u.max_cell_dimension();

  std::pair<std::size_t, std::size_t> ownership_range =
    Fu.function_space()->dofmap()->ownership_range();
  const std::size_t offset = Fu.function_space()->dofmap()->ownership_range().first;
  std::pair<std::size_t, std::size_t> rows = A.local_range(0);

  for (std::size_t row = rows.first; row < rows.second; ++row)
  {
    std::size_t ii = row - offset;
    std::vector<std::size_t> cols;

    std::vector<double> bij;
    A.getrow(row, cols, bij);
    
    std::vector<double> alpha_ij(cols.size(), 0.0);

    double bij_udiff = 0.0;
    double bij_udiff_max = 0.0;

    for (std::size_t col = 0; col < cols.size(); ++col)
    {
      std::size_t jj = cols[col] - offset;
      if (jj + offset >= rows.first && jj + offset < rows.second)
      {
        if (ii != jj)
        {
	  // double const Sj = 0.5 * Fu_arr[jj] * Fu_arr[jj];
	  // double const Si = 0.5 * Fu_arr[ii] * Fu_arr[ii];
	  double const Sj = Fu_arr[jj];
	  double const Si = Fu_arr[ii];
	  
	  double diff_u = Sj - Si;

	  bij_udiff += bij[col] * diff_u;
	  bij_udiff_max += fabs(bij[col]) * fabs(diff_u);
	}
      }
    }

    double const alpha = fabs(bij_udiff) / std::max(bij_udiff_max, 1e-6);

    /// ReLU
    // double const alpha0 = 0.5;
    // double const q = 4.0;
    // double const psi = std::max(0.0, (alpha - alpha0)/(1.0 - alpha0));
    // double const psi2 = psi*psi;
    // Falpha_arr[ii] = psi; 

    // /// sigmoid
    // double const s = 20.0;
    // double const x0 = 0.5;
    // double const psi = 1.0/(1.0 + std::exp(-s*(alpha-x0)));

    /// linear
    double const psi = alpha;
    
    Falpha_arr[ii] = psi; 
    
  }
  Falpha.vector()->set_local(Falpha_arr);
  Falpha.vector()->apply("insert");
};

//-----------------------------------------------------------------------------
void apply_smoothing(Matrix const & A, Function & Fu)
{
  std::vector<double> Fu_arr(Fu.vector()->local_size());
  std::vector<double> Fu_s_arr(Fu.vector()->local_size());
  Fu.vector()->get_local(Fu_arr);

  ufc::cell ufc_cell;
  std::vector<double> vertex_coordinates0;

  const GenericDofMap& dofmap_u = *Fu.function_space()->dofmap();
  dolfin::la_index dofsize_u = dofmap_u.max_cell_dimension();

  std::pair<std::size_t, std::size_t> ownership_range =
    Fu.function_space()->dofmap()->ownership_range();
  const std::size_t offset = Fu.function_space()->dofmap()->ownership_range().first;
  std::pair<std::size_t, std::size_t> rows = A.local_range(0);

  for (std::size_t row = rows.first; row < rows.second; ++row)
  {
    std::size_t ii = row - offset;
    std::vector<std::size_t> cols;

    std::vector<double> bij;
    A.getrow(row, cols, bij);

    double sum = 0.0;
    
    for (std::size_t col = 0; col < cols.size(); ++col)
    {
      std::size_t jj = cols[col] - offset;
      if (jj + offset >= rows.first && jj + offset < rows.second)
      {
        if (ii != jj)
        {
	  sum += Fu_arr[jj];
	}
      }

      double const cols_size = (double)(cols.size());
      double const d = cols_size - 1.0;
      
      // Fu_s_arr[ii] = (sum + d * Fu_arr[ii]) / (2.0*d);

      double const l = 4.0;
      Fu_s_arr[ii] = (sum + (l-1.0)*d*Fu_arr[ii]) / (l*d);
      
    }
  }
  Fu.vector()->set_local(Fu_s_arr);
  Fu.vector()->apply("insert");
};

//------------------------------------------------------------------------
int main(int argc, char *argv[])
{
  parameters["ghost_mode"] = "shared_facet";

  Point point0(xmin, ymin);
  Point point1(xmax, ymax);
  int N = 100;

  for (int i = 0; i < 1; ++i)
    {
  
      auto mesh = std::make_shared<RectangleMesh>(point0, point1, N, N);
      auto mesh_fine = std::make_shared<RectangleMesh>(point0, point1, N, N);

      /// Parameters for time-stepping
      const double N_sample = 10.0;
      const double T = 0.5;
      const double hmin = mesh->hmin();
      const double kk = 0.25 * hmin;
      const std::size_t NN = static_cast<std::size_t>(T / kk);
      const double k = T / static_cast<double>(NN);
      double t = 0.0;

      /// define constants
      auto Ck = std::make_shared<Constant>(k);

      // Create periodic boundary condition
      auto periodic_boundary = std::make_shared<PeriodicBoundaryXY1>();

      // Create function space
      auto V = std::make_shared<Burger::Form_a_FunctionSpace_0>(mesh);//, periodic_boundary);
      auto V3 = std::make_shared<Burger::CoefficientSpace_u_ex>(mesh);//, periodic_boundary);
      // auto V = std::make_shared<Burger::Form_a_FunctionSpace_0>(mesh, periodic_boundary);
      // auto V3 = std::make_shared<Burger::CoefficientSpace_u_ex>(mesh, periodic_boundary);
    
      // Source term and initial condition
      auto u = std::make_shared<Function>(V);
      auto u0 = std::make_shared<Function>(V);
      auto u_ex = std::make_shared<Function>(V3);
      auto uerr = std::make_shared<Function>(V);
      auto Feps = std::make_shared<Function>(V);
      auto Falpha = std::make_shared<Function>(V);
  
      // /// construct the initial condition
      // auto uin = std::make_shared<Indata>();
      // u0->interpolate(*uin);
      // u_ex->interpolate(*uin);

      /// initial condition for Burger
      auto burger_exact = std::make_shared<Burger_Exact_sol>(t);
      u0->interpolate(*burger_exact);
      u_ex->interpolate(*burger_exact);
      auto boundary = std::make_shared<DirichletBoundary>();
      DirichletBC bc(V, u_ex, boundary);
      
      Burger::Form_a a(V, V);
      Burger::Form_test test(V, V);
      Burger::Form_L L(V);
      L.u0 = u0;
      L.k  = Ck;
      L.alpha  = Falpha;
      //
      a.k  = Ck;
      a.alpha  = Falpha;
      a.u0  = u0;
  
      ///
      std::shared_ptr<Matrix> A(new Matrix);
      Vector b;
      assemble(*A, a);
      assemble(b, L);

      // std::ofstream matrix_out("matrix_cpp.txt");
      // if (matrix_out.is_open())
      // {
      //     for (int row = 0; row < A->size(0); ++row)  // Iterate over rows
      //     {
      //         std::vector<size_t> cols;        // Column indices for non-zero entries
      //         std::vector<double> values;  // Corresponding non-zero values

      //         A->getrow(row, cols, values); // Extract the row entries
      //         for (size_t idx = 0; idx < cols.size(); ++idx)
      //         {
      //             matrix_out << row << " " << cols[idx] << " " << values[idx] << "\n";
      //         }
      //     }
      //     matrix_out.close(); // Close the file
      //     std::cout << "Matrix saved to matrix_cpp.txt" << std::endl;
      // }
      // else
      // {
      //     std::cerr << "Error: Could not open file matrix_cpp.txt for writing." << std::endl;
      // }

      // Krylov solver
      LinearSolver ksolver("direct");

      /// set initial condition
      *u->vector() = *u0->vector();

      compute_alphaij(*A, *u, *Falpha);

      File fileex("results/exact.pvd","compressed");
      File fileu("results/u.pvd","compressed");
      File filea("results/alpha.pvd","compressed");
      fileu << std::pair<const Function*, double>(u.get(), t);
      filea << std::pair<const Function*, double>(Falpha.get(), t);


      double time_save = 0.0;
      t = 0.0;
      // Time-stepping
      Progress prog("Time-stepping");
      while (t < T - k/2. )
      {
        tic();

        *u0->vector() = *u->vector();
        
        // Assemble vector and apply boundary conditions
        assemble(*A, a);
        assemble(b,  L);
        bc.apply(*A, b);

        // Solve the linear system 
        ksolver.solve(*A, *u->vector(), b);

        /// make solution a bit smoother
        // apply_smoothing(*A, *u);

        /// only for Burger
        u_ex->interpolate(*burger_exact);

        compute_alphaij(*A, *u, *Falpha);

        if (time_save > T/N_sample || time_save > T - DOLFIN_EPS)
          {
            // Save solution in VTK format
            fileu << std::pair<const Function*, double>(u.get(), t);
            filea << std::pair<const Function*, double>(Falpha.get(), t);
            time_save = 0.0;
          }
        
        // Move to next interval
        // prog = t/T;
        t += k;
        time_save += k;

        *uerr->vector()  = *u->vector();
        *uerr->vector() -= *u0->vector();
        double const ue_l2 = uerr->vector()->norm("linf");
        std::cout << "t = " << t
            << ", dt = " << k
            << ", l2 = " << ue_l2
            << ", iter took " << toc() << " sec"
            << std::endl;
        if (ue_l2 > 1e3)
          error("shit happened");
        
      }

      fileex << std::pair<const Function*, double>(u_ex.get(), t);
      fileu << std::pair<const Function*, double>(u.get(), t);
      filea << std::pair<const Function*, double>(Falpha.get(), t);

      /// compute the error norm
      Burger::Form_L1 L1(mesh, u, u_ex);
      Burger::Form_L2 L2(mesh, u, u_ex);
      /// save to file
      std::ostringstream error_fname;
      error_fname << "res.m";
      std::ofstream errorFile;
      errorFile.open((error_fname.str()).c_str(), std::ios::app);
      errorFile << V->dofmap()->global_dimension() << "\t"
		<< assemble(L1) << "\t"
		<< sqrt(assemble(L2)) << "\n";

      N = N * 2;
    }
  return 0;
}

