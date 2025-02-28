// # solidmechanics
//
// Solve a compressible neo-Hookean model in 3D.

// ## UFL form file
//
// The UFL file is implemented in
// {download}`demo_solidmechanics/solidmechanics.py`.
// ````{admonition} UFL form implemented in python
// :class: dropdown
// ![ufl-code]
// ````
//

// ## C++ program

#include "solidmechanics.h"
#include <algorithm>
#include <basix/finite-element.h>
#include <basix/quadrature.h>
#include <dolfinx.h>
#include <dolfinx/common/log.h>
#include <dolfinx/fem/FiniteElement.h>
#include <dolfinx/fem/petsc.h>

using namespace dolfinx;
using T = PetscScalar;
using U = typename dolfinx::scalar_value_type_t<T>;

std::shared_ptr<fem::FunctionSpace<U>>
quadrature_functionspace(std::shared_ptr<mesh::Mesh<U>> _mesh,
                         std::vector<std::size_t> value_shape) {

  mesh::CellType cell_type = mesh::CellType::hexahedron;

  auto quadrature = basix::quadrature::make_quadrature<U>(
      basix::quadrature::type::Default, basix::cell::type::hexahedron,
      basix::polyset::type::standard, 2);

  std::span<const T> pts = quadrature.front();
  // std::span<const T> weights = quadrature.back();

  std::array<std::size_t, 2> pshape = {pts.size() / 3, 3};

  std::size_t block_size =
      value_shape.empty()
          ? 1
          : std::accumulate(value_shape.begin(), value_shape.end(), 1,
                            std::multiplies{});

  auto Qe = std::make_shared<const fem::FiniteElement<T>>(
      fem::FiniteElement(cell_type, pts, pshape, value_shape, false));

  fem::ElementDofLayout layout(block_size, Qe->entity_dofs(),
                               Qe->entity_closure_dofs(), {}, {});

  std::shared_ptr<const fem::DofMap> dofmap =
      std::make_shared<const fem::DofMap>(create_dofmap(
          _mesh->comm(), layout, *_mesh->topology(), nullptr, nullptr));

  return std::make_shared<fem::FunctionSpace<U>>(
      fem::FunctionSpace(_mesh, Qe, dofmap));
}

int main(int argc, char *argv[]) {
  init_logging(argc, argv);
  PetscInitialize(&argc, &argv, nullptr, nullptr);

  // Set the logging thread name to show the process rank
  int mpi_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  std::string fmt =
      "[%Y-%m-%d %H:%M:%S.%e] [RANK " + std::to_string(mpi_rank) + "] [%l] %v";
  spdlog::set_pattern(fmt);
  {

    // Create mesh and define function space
    auto mesh = std::make_shared<mesh::Mesh<U>>(mesh::create_box<U>(
        MPI_COMM_WORLD, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {1, 1, 1},
        mesh::CellType::hexahedron,
        mesh::create_cell_partitioner(mesh::GhostMode::none)));

    auto element = basix::create_element<U>(
        basix::element::family::P, basix::cell::type::hexahedron, 2,
        basix::element::lagrange_variant::unset,
        basix::element::dpc_variant::unset, false);

    auto V =
        std::make_shared<fem::FunctionSpace<U>>(fem::create_functionspace<U>(
            mesh, std::make_shared<fem::FiniteElement<U>>(
                      element, std::vector<std::size_t>{3})));

    auto Q6 = quadrature_functionspace(mesh, {6});
    auto Q36 = quadrature_functionspace(mesh, {36});

    // Define solution function
    auto u = std::make_shared<fem::Function<T>>(V);

    auto stress = std::make_shared<fem::Function<T>>(Q6);
    auto tangent = std::make_shared<fem::Function<T>>(Q36);

    fem::Form<T> a =
        fem::create_form<T>(*form_solidmechanics_a, {V, V},
                            {{"u", u}, {"tangent", tangent}}, {}, {}, {});
    fem::Form<T> L = fem::create_form<T>(*form_solidmechanics_L, {V},
                                         {{"stress", stress}}, {}, {}, {});

    la::petsc::Matrix A(fem::petsc::create_matrix(a), false);

    MatZeroEntries(A.mat());
    fem::assemble_matrix(la::petsc::Matrix::set_block_fn(A.mat(), ADD_VALUES),
                         a, {});
    MatAssemblyBegin(A.mat(), MAT_FLUSH_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FLUSH_ASSEMBLY);
    fem::set_diagonal<T>(la::petsc::Matrix::set_fn(A.mat(), INSERT_VALUES), *V,
                         {});
    MatAssemblyBegin(A.mat(), MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(A.mat(), MAT_FINAL_ASSEMBLY);

    std::string name = "matrix.txt";
    PetscViewer viewer;
    PetscViewerASCIIOpen(MPI_COMM_WORLD, name.c_str(), &viewer);
    PetscViewerPushFormat(viewer, PETSC_VIEWER_ASCII_DENSE);
    MatView(A.mat(), viewer);
    PetscViewerPopFormat(viewer);
    PetscViewerDestroy(&viewer);
  }

  PetscFinalize();

  return 0;
}