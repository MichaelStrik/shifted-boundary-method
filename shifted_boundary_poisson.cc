#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/data_out.h>
 
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/base/smartpointer.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparse_direct.h>
 
#include <array>
#include <fstream>
#include <iostream>
 
namespace Step7
{
  using namespace dealii;
 
 
  template <int dim>
  class SolutionBase
  {
  protected:
    static const std::array<Point<dim>, 3> source_centers;
    static const double                    width;
  };
 
 
  template <>
  const std::array<Point<1>, 3> SolutionBase<1>::source_centers = {
    {Point<1>(-1.0 / 3.0), Point<1>(0.0), Point<1>(+1.0 / 3.0)}};
 
  template <>
  const std::array<Point<2>, 3> SolutionBase<2>::source_centers = {
    {Point<2>(-0.5, +0.5), Point<2>(-0.5, -0.5), Point<2>(+0.5, -0.5)}};
 
  template <int dim>
  const double SolutionBase<dim>::width = 1. / 8.;
 
 
 
  template <int dim>
  class Solution : public Function<dim>, protected SolutionBase<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
 
    virtual Tensor<1, dim>
    gradient(const Point<dim> & p,
             const unsigned int component = 0) const override;
  };
 
 
  template <int dim>
  double Solution<dim>::value(const Point<dim> &p, const unsigned int) const
  {
    const double pi = std::acos(-1);
    return std::sin(2.0 * pi * p[0]);
  }
 
 
  template <int dim>
  Tensor<1, dim> Solution<dim>::gradient(const Point<dim> &p,
                                         const unsigned int) const
  {
    const double pi = std::acos(-1);
    double arr[2];
    arr[0] = 2.0 * pi * std::cos(2.0 * pi * p[0]);
    arr[1] = 0.;
    Tensor<1, dim> return_value(arr);
    return return_value;
  }
 
 
 
  template <int dim>
  class RightHandSide : public Function<dim>, protected SolutionBase<dim>
  {
  public:
    virtual double value(const Point<dim> & p,
                         const unsigned int component = 0) const override;
  };
 
 
  template <int dim>
  double RightHandSide<dim>::value(const Point<dim> &p,
                                   const unsigned int) const
  {
    const double pi = std::acos(-1);
    return (4.0 * pi * pi) * std::sin(2.0 * pi * p[0]);
  }
 
 
 
  template <int dim>
  class HelmholtzProblem
  {
  public:
    enum RefinementMode
    {
      global_refinement,
      adaptive_refinement
    };
 
    HelmholtzProblem(const FiniteElement<dim> &fe,
                     const RefinementMode      refinement_mode);
 
    void run();
 
  private:
    void setup_system();
    const Tensor<1,dim> distance_vector_to_boundary(int boundary_id);
    void assemble_system();
    void solve();
    void refine_grid();
    void process_solution(const unsigned int cycle);
  
 
    Triangulation<dim> triangulation;
    DoFHandler<dim>    dof_handler;
 
    SmartPointer<const FiniteElement<dim>> fe;
 
    AffineConstraints<double> hanging_node_constraints;
 
    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;
 
    Vector<double> solution;
    Vector<double> system_rhs;
 
    const RefinementMode refinement_mode;
 
    ConvergenceTable convergence_table;
  };
 
  template <int dim>
  HelmholtzProblem<dim>::HelmholtzProblem(const FiniteElement<dim> &fe,
                                          const RefinementMode refinement_mode)
    : dof_handler(triangulation)
    , fe(&fe)
    , refinement_mode(refinement_mode)
  {}
 
 
  template <int dim>
  void HelmholtzProblem<dim>::setup_system()
  {
    dof_handler.distribute_dofs(*fe);
    DoFRenumbering::Cuthill_McKee(dof_handler);
 
    hanging_node_constraints.clear();
    DoFTools::make_hanging_node_constraints(dof_handler,
                                            hanging_node_constraints);
    hanging_node_constraints.close();
 
    DynamicSparsityPattern dsp(dof_handler.n_dofs(), dof_handler.n_dofs());
    DoFTools::make_sparsity_pattern(dof_handler, dsp);
    hanging_node_constraints.condense(dsp);
    sparsity_pattern.copy_from(dsp);
 
    system_matrix.reinit(sparsity_pattern);
 
    solution.reinit(dof_handler.n_dofs());
    system_rhs.reinit(dof_handler.n_dofs());
  }
 
  template <int dim>
  const Tensor<1,dim> HelmholtzProblem<dim>::distance_vector_to_boundary(int boundary_id) {
    Tensor<1,dim> distance_vector;

    if (boundary_id == 1)
    {
      distance_vector[0] = -1;
    }
    else if (boundary_id == 2)
    {
      distance_vector[0] =  1;
    }

    return distance_vector;
  }
 
  // Nitsche method
  /* template <int dim>
  void HelmholtzProblem<dim>::assemble_system()
  { 

    double alpha = (fe->degree +1) * (fe->degree);

    QGauss<dim>     quadrature_formula(fe->degree + 1);
    QGauss<dim - 1> face_quadrature_formula(fe->degree + 1);
 
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
 
    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
 
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
 
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
    FEValues<dim> fe_values(*fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
 
    FEFaceValues<dim> fe_face_values(*fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values | update_gradients);
 
    RightHandSide<dim>  right_hand_side;
    std::vector<double> rhs_values(n_q_points);
 
    Solution<dim> exact_solution;
 
    // cell iteration
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0.;
        cell_rhs    = 0.;

        fe_values.reinit(cell);
 
        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   rhs_values);

        const double cell_side_length = cell->minimum_vertex_distance();
        // quadrature point iteration on cell
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point){
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j){
                cell_matrix(i, j) +=
                  (fe_values.shape_grad(i, q_point) *     // grad phi_i(x_q)
                      fe_values.shape_grad(j, q_point) *   // grad phi_j(x_q)
                   fe_values.JxW(q_point));                // dx
              }
 
              cell_rhs(i) += (fe_values.shape_value(i, q_point) * // phi_i(x_q)
                              rhs_values[q_point] *               // f(x_q)
                              fe_values.JxW(q_point));            // dx
            }
        }
        // face iteration on cell
        for (const auto &face : cell->face_iterators()){
          if (face->at_boundary()){
              fe_face_values.reinit(cell, face);
 
              for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point){
                for (unsigned int i = 0; i < dofs_per_cell; ++i){
                  for(unsigned int j = 0; j < dofs_per_cell; ++j){
                    cell_matrix(i,j) -=
                        (fe_face_values.shape_value(i, q_point) *
                        fe_face_values.shape_grad(j, q_point)*
                        fe_face_values.normal_vector(q_point)*
                        fe_face_values.JxW(q_point));

                    cell_matrix(i,j) -=
                        (fe_face_values.shape_grad(i, q_point)*
                        fe_face_values.normal_vector(q_point)*
                        fe_face_values.shape_value(j, q_point)*
                        fe_face_values.JxW(q_point));

                    cell_matrix(i,j) +=
                        ((alpha/cell_side_length) *
                        fe_face_values.shape_value(i, q_point) *
                        (fe_face_values.shape_value(j, q_point)) *
                        fe_face_values.JxW(q_point));
                    }
                    
                  cell_rhs(i) += ((alpha/cell_side_length) *
                                  exact_solution.value(fe_face_values.quadrature_point(q_point))*
                                  fe_face_values.shape_value(i, q_point)*
                                  fe_face_values.JxW(q_point));

                  cell_rhs(i) -= 
                        (fe_face_values.shape_grad(i, q_point)*
                        fe_face_values.normal_vector(q_point)*
                        exact_solution.value(fe_face_values.quadrature_point(q_point))*
                        fe_face_values.JxW(q_point));
                  }
                }
            }
        }
 
        // write system matrix and rhs
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              system_matrix.add(local_dof_indices[i],
                                local_dof_indices[j],
                                cell_matrix(i, j));
 
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }
  } */
 
  // shifted boundary method
  template <int dim>
  void HelmholtzProblem<dim>::assemble_system()
  { 

    double alpha = (fe->degree +1) * (fe->degree);

    QGauss<dim>     quadrature_formula(fe->degree + 1);
    QGauss<dim - 1> face_quadrature_formula(fe->degree + 1);
 
    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature_formula.size();
 
    const unsigned int dofs_per_cell = fe->n_dofs_per_cell();
 
    FullMatrix<double> cell_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     cell_rhs(dofs_per_cell);
 
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
 
    FEValues<dim> fe_values(*fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
 
    FEFaceValues<dim> fe_face_values(*fe,
                                     face_quadrature_formula,
                                     update_values | update_quadrature_points |
                                       update_normal_vectors |
                                       update_JxW_values | update_gradients);
 
    RightHandSide<dim>  right_hand_side;
    std::vector<double> rhs_values(n_q_points);
 
    Solution<dim> exact_solution;
 
    // cell iteration
    for (const auto &cell : dof_handler.active_cell_iterators())
      {
        cell_matrix = 0.;
        cell_rhs    = 0.;

        fe_values.reinit(cell);
 
        right_hand_side.value_list(fe_values.get_quadrature_points(),
                                   rhs_values);

        const double cell_side_length = cell->minimum_vertex_distance();
        // quadrature point iteration on cell
        for (unsigned int q_point = 0; q_point < n_q_points; ++q_point){
          for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              for (unsigned int j = 0; j < dofs_per_cell; ++j){
                cell_matrix(i, j) +=
                  (fe_values.shape_grad(i, q_point) *     // grad phi_i(x_q)
                      fe_values.shape_grad(j, q_point) *   // grad phi_j(x_q)
                   fe_values.JxW(q_point));                // dx
              }
 
              cell_rhs(i) += (fe_values.shape_value(i, q_point) * // phi_i(x_q)
                              rhs_values[q_point] *               // f(x_q)
                              fe_values.JxW(q_point));            // dx
            }
        }
        // face iteration on cell
        for (const auto &face : cell->face_iterators()){
          if (face->at_boundary()){
              fe_face_values.reinit(cell, face);
              const int boundary_id = face->boundary_id();
 
              for (unsigned int q_point = 0; q_point < n_face_q_points; ++q_point){
                for (unsigned int i = 0; i < dofs_per_cell; ++i){
                  for(unsigned int j = 0; j < dofs_per_cell; ++j){
                    cell_matrix(i,j) -=
                        ((fe_face_values.shape_value(i, q_point)+     // phi_i(x_q)
                          fe_face_values.shape_grad(i, q_point)*      // grad phi_i(x_q)
                          distance_vector_to_boundary(boundary_id))* // d(x_q)
                        fe_face_values.shape_grad(j, q_point)*      // grad phi_j(x_q)
                        fe_face_values.normal_vector(q_point)*      // n(x_q)
                        fe_face_values.JxW(q_point));               // Jacobian (trafo)

                    cell_matrix(i,j) -=
                        (fe_face_values.shape_grad(i, q_point)*     // grad phi_i(x_q)
                        fe_face_values.normal_vector(q_point)*      // n(x_q)
                        ( fe_face_values.shape_value(j, q_point)+     // phi_j(x_q)
                          fe_face_values.shape_grad(j, q_point)*      // grad phi_j(x_q)
                          distance_vector_to_boundary(boundary_id))*   // d(x_q)
                        fe_face_values.JxW(q_point));

                    cell_matrix(i,j) +=
                        (fe_face_values.shape_grad(i, q_point)*    // grad phi_i(x_q)
                        distance_vector_to_boundary(boundary_id)*  // d(x_q)
                        fe_face_values.shape_grad(j, q_point)*     // grad phi_j(x_q)
                        fe_face_values.normal_vector(q_point)*      // n(x_q)
                        fe_face_values.JxW(q_point));


                    cell_matrix(i,j) +=
                        ((alpha/cell_side_length) *                 // penalty/Nitsche-parameter over h
                        ( fe_face_values.shape_value(i, q_point)+     // phi_i(x_q)
                          fe_face_values.shape_grad(i, q_point)*      // grad phi_i(x_q)
                          distance_vector_to_boundary(boundary_id))*  // d(x_q)
                        ( fe_face_values.shape_value(j, q_point)+     // phi_j(x_q)
                          fe_face_values.shape_grad(j, q_point)*      // grad phi_j(x_q)
                          distance_vector_to_boundary(boundary_id))*  // d(x_q)
                        fe_face_values.JxW(q_point));
                    
                    }
                    
                  
                  cell_rhs(i) -= 
                        (fe_face_values.shape_grad(i, q_point)*   //phi_i(x_q)
                        fe_face_values.normal_vector(q_point)*    // n(x_q)
                        exact_solution.value(fe_face_values.quadrature_point(q_point))* // u_D(x_q) (Dirichlet boundary function)
                        fe_face_values.JxW(q_point));
                  

                  cell_rhs(i) += ((alpha/cell_side_length) *    // penalty/Nitsche-parameter
                                  ( fe_face_values.shape_value(i, q_point)+     // phi_i(x_q)
                                    fe_face_values.shape_grad(i, q_point)*      // grad phi_i(x_q)
                                    distance_vector_to_boundary(boundary_id))*  // d(x_q)
                                  exact_solution.value(fe_face_values.quadrature_point(q_point))* // u_D(x_q) (Dirichlet boundary function)
                                  fe_face_values.JxW(q_point));
                  }
                }
            }
        }
 
        // write system matrix and rhs
        cell->get_dof_indices(local_dof_indices);
        for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            for (unsigned int j = 0; j < dofs_per_cell; ++j)
              system_matrix.add(local_dof_indices[i],
                                local_dof_indices[j],
                                cell_matrix(i, j));
 
            system_rhs(local_dof_indices[i]) += cell_rhs(i);
          }
      }
  } 
 
  template <int dim>
  void HelmholtzProblem<dim>::solve()
  {

    SparseDirectUMFPACK solver;
    solver.initialize(system_matrix);
    solver.vmult(solution,system_rhs);


    // SolverControl solver_control(1000, 1e-12);
    // SolverCG<>    cg(solver_control);

    // PreconditionSSOR<> preconditioner;
    // preconditioner.initialize(system_matrix, 1.2);

    // cg.solve(system_matrix, solution, system_rhs, preconditioner);

    // hanging_node_constraints.distribute(solution);

  }
 
 
 
  template <int dim>
  void HelmholtzProblem<dim>::refine_grid()
  {
    switch (refinement_mode)
      {
        case global_refinement:
          {
            triangulation.refine_global(1);
            break;
          }
 
        case adaptive_refinement:
          {
            Vector<float> estimated_error_per_cell(
              triangulation.n_active_cells());
 
            KellyErrorEstimator<dim>::estimate(
              dof_handler,
              QGauss<dim - 1>(fe->degree + 1),
              std::map<types::boundary_id, const Function<dim> *>(),
              solution,
              estimated_error_per_cell);
 
            GridRefinement::refine_and_coarsen_fixed_number(
              triangulation, estimated_error_per_cell, 0.3, 0.03);
 
            triangulation.execute_coarsening_and_refinement();
 
            break;
          }
 
        default:
          {
            Assert(false, ExcNotImplemented());
          }
      }
  }
 
 
 
  template <int dim>
  void HelmholtzProblem<dim>::process_solution(const unsigned int cycle)
  {
    Vector<float> difference_per_cell(triangulation.n_active_cells());
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(fe->degree + 1),
                                      VectorTools::L2_norm);
    const double L2_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::L2_norm);
 
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      QGauss<dim>(fe->degree + 1),
                                      VectorTools::H1_seminorm);
    const double H1_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::H1_seminorm);
 
    const QTrapezoid<1>  q_trapez;
    const QIterated<dim> q_iterated(q_trapez, fe->degree * 2 + 1);
    VectorTools::integrate_difference(dof_handler,
                                      solution,
                                      Solution<dim>(),
                                      difference_per_cell,
                                      q_iterated,
                                      VectorTools::Linfty_norm);
    const double Linfty_error =
      VectorTools::compute_global_error(triangulation,
                                        difference_per_cell,
                                        VectorTools::Linfty_norm);
 
    const unsigned int n_active_cells = triangulation.n_active_cells();
    const unsigned int n_dofs         = dof_handler.n_dofs();
 
    std::cout << "Cycle " << cycle << ':' << std::endl
              << "   Number of active cells:       " << n_active_cells
              << std::endl
              << "   Number of degrees of freedom: " << n_dofs << std::endl;
 
    convergence_table.add_value("cycle", cycle);
    convergence_table.add_value("cells", n_active_cells);
    convergence_table.add_value("dofs", n_dofs);
    convergence_table.add_value("L2", L2_error);
    convergence_table.add_value("H1", H1_error);
    convergence_table.add_value("Linfty", Linfty_error);
  }
 
 
 
  template <int dim>
  void HelmholtzProblem<dim>::run()
  {
    const unsigned int n_cycles =
      (refinement_mode == global_refinement) ? 5 : 5;
    for (unsigned int cycle = 0; cycle < n_cycles; ++cycle)
      {
        if (cycle == 0)
          {
            GridGenerator::hyper_cube(triangulation, -1., 1.);
            triangulation.refine_global(3);

            for (const auto &cell : triangulation.cell_iterators())
              for (const auto &face : cell->face_iterators())
                {
                  const auto center = face->center();
                  
                  
                  if (std::fabs(center(0) - (-1.0)) < 1e-12){
                    // left boundary
                    face->set_boundary_id(1);
                  }

                  else if (std::fabs(center(0) - (1.0)) < 1e-12) {
                    // right boundary
                    face->set_boundary_id(2);
                  }

                  else if (std::fabs(center(1) - (-1.0)) < 1e-12) {
                    // bottom boundary
                    face->set_boundary_id(3);
                  }

                  else if (std::fabs(center(1) - (1.0)) < 1e-12) {
                    // top boundary
                    face->set_boundary_id(4);
                  }
                  
                }
          }
        else
          refine_grid();
 
 
        setup_system();
 
        assemble_system();
        solve();
 
        process_solution(cycle);
      }
 
 
    std::string vtk_filename;
    switch (refinement_mode)
      {
        case global_refinement:
          vtk_filename = "solution-global";
          break;
        case adaptive_refinement:
          vtk_filename = "solution-adaptive";
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
 
    switch (fe->degree)
      {
        case 1:
          vtk_filename += "-q1";
          break;
        case 2:
          vtk_filename += "-q2";
          break;
 
        default:
          Assert(false, ExcNotImplemented());
      }
 
    vtk_filename += ".vtk";
    std::ofstream output(vtk_filename);
 
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(solution, "solution");
 
    data_out.build_patches(fe->degree);
    data_out.write_vtk(output);
 
 
 
    convergence_table.set_precision("L2", 3);
    convergence_table.set_precision("H1", 3);
    convergence_table.set_precision("Linfty", 3);
 
    convergence_table.set_scientific("L2", true);
    convergence_table.set_scientific("H1", true);
    convergence_table.set_scientific("Linfty", true);
 
    convergence_table.set_tex_caption("cells", "\\# cells");
    convergence_table.set_tex_caption("dofs", "\\# dofs");
    convergence_table.set_tex_caption("L2", "@f$L^2@f$-error");
    convergence_table.set_tex_caption("H1", "@f$H^1@f$-error");
    convergence_table.set_tex_caption("Linfty", "@f$L^\\infty@f$-error");
 
    convergence_table.set_tex_format("cells", "r");
    convergence_table.set_tex_format("dofs", "r");
 
    std::cout << std::endl;
    convergence_table.write_text(std::cout);
 
    std::string error_filename = "error";
    switch (refinement_mode)
      {
        case global_refinement:
          error_filename += "-global";
          break;
        case adaptive_refinement:
          error_filename += "-adaptive";
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
 
    switch (fe->degree)
      {
        case 1:
          error_filename += "-q1";
          break;
        case 2:
          error_filename += "-q2";
          break;
        default:
          Assert(false, ExcNotImplemented());
      }
 
    error_filename += ".tex";
    std::ofstream error_table_file(error_filename);
 
    convergence_table.write_tex(error_table_file);
 
 
 
    if (refinement_mode == global_refinement)
      {
        convergence_table.add_column_to_supercolumn("cycle", "n cells");
        convergence_table.add_column_to_supercolumn("cells", "n cells");
 
        std::vector<std::string> new_order;
        new_order.emplace_back("n cells");
        new_order.emplace_back("H1");
        new_order.emplace_back("L2");
        convergence_table.set_column_order(new_order);
 
        convergence_table.evaluate_convergence_rates(
          "L2", ConvergenceTable::reduction_rate);
        convergence_table.evaluate_convergence_rates(
          "L2", ConvergenceTable::reduction_rate_log2);
        convergence_table.evaluate_convergence_rates(
          "H1", ConvergenceTable::reduction_rate);
        convergence_table.evaluate_convergence_rates(
          "H1", ConvergenceTable::reduction_rate_log2);
 
        std::cout << std::endl;
        convergence_table.write_text(std::cout);
 
        std::string conv_filename = "convergence";
        switch (refinement_mode)
          {
            case global_refinement:
              conv_filename += "-global";
              break;
            case adaptive_refinement:
              conv_filename += "-adaptive";
              break;
            default:
              Assert(false, ExcNotImplemented());
          }
        switch (fe->degree)
          {
            case 1:
              conv_filename += "-q1";
              break;
            case 2:
              conv_filename += "-q2";
              break;
            default:
              Assert(false, ExcNotImplemented());
          }
        conv_filename += ".tex";
 
        std::ofstream table_file(conv_filename);
        convergence_table.write_tex(table_file);
      }
  }
 
} // namespace Step7
 
 
int main()
{
  const unsigned int dim = 2;
 
  try
    {
      using namespace dealii;
      using namespace Step7;
 
      {
        std::cout << "Solving with Q1 elements, global refinement" << std::endl
                  << "===========================================" << std::endl
                  << std::endl;
 
        FE_Q<dim>             fe(1);
        HelmholtzProblem<dim> helmholtz_problem_2d(
          fe, HelmholtzProblem<dim>::global_refinement);
 
        helmholtz_problem_2d.run();
 
        std::cout << std::endl;
      }
 
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
 
  return 0;
}