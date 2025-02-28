// ```text
// Copyright (C) 2024 Jack S. Hale and Garth N. Wells
// This file is part of DOLFINx (https://www.fenicsproject.org)
// SPDX-License-Identifier:    LGPL-3.0-or-later
// ```

// # Custom cell kernel assembly
//
// This demo shows various methods to define custom cell kernels in C++
// and have them assembled into DOLFINx linear algebra data structures.

#include <basix/finite-element.h>
#include <basix/indexing.h>
#include <basix/mdspan.hpp>
#include <basix/quadrature.h>
#include <cmath>
#include <concepts>
#include <dolfinx.h>
#include <dolfinx/la/MatrixCSR.h>
#include <dolfinx/la/SparsityPattern.h>
#include <functional>
#include <stdint.h>
#include <utility>
#include <vector>

using namespace dolfinx;

template <typename T, std::size_t ndim>
using mdspand_t = MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<
    T, MDSPAN_IMPL_STANDARD_NAMESPACE::dextents<std::size_t, ndim>>;
template <typename T, std::size_t n0, std::size_t n1>
using mdspan2_t =
    MDSPAN_IMPL_STANDARD_NAMESPACE::mdspan<T,
                                           std::extents<std::size_t, n0, n1>>;

/// @brief Compute the P1 element mass matrix on the reference cell.
/// @tparam T Scalar type.
/// @param phi Basis functions.
/// @param w Integration weights.
/// @return Element reference matrix (row-major storage).
template <typename T>
std::array<T, 576> A_ref(mdspand_t<const T, 4> phi, std::span<const T> w) {
  std::array<T, 576> A_b{};

  mdspan2_t<T, 24, 24> A(A_b.data());

  for (std::size_t k = 0; k < phi.extent(1); ++k)   // quadrature point
    for (std::size_t i = 0; i < A.extent(0); ++i)   // row i
      for (std::size_t j = 0; j < A.extent(1); ++j) // column j
        A(i, j) += w[k] * phi(0, k, i, 0) * phi(0, k, j, 0);
  // return A_b;
  return A_b;
}

/// @brief Assemble a matrix operator using a lambda kernel function.
///
/// The lambda function can be inlined in the assembly code, which can
/// be important for performance for lightweight kernels.
///
/// @tparam T Scalar type.
/// @param g mesh geometry.
/// @param dofmap dofmap.
/// @param kernel Element kernel to execute.
/// @param cells Cells to execute the kernel over.
/// @return Frobenius norm squared of the matrix.
template <std::floating_point T>
double assemble_matrix1(const mesh::Geometry<T> &g, const fem::DofMap &dofmap,
                        auto kernel, std::span<const std::int32_t> cells) {
  auto sp = la::SparsityPattern(dofmap.index_map->comm(),
                                {dofmap.index_map, dofmap.index_map},
                                {dofmap.index_map_bs(), dofmap.index_map_bs()});

  fem::sparsitybuild::cells(sp, {cells, cells}, {dofmap, dofmap});
  sp.finalize();
  la::MatrixCSR<T> A(sp);
  auto ident = [](auto, auto, auto, auto) {}; // DOF permutation not required
  common::Timer timer("Assembler1 lambda (matrix)");

  //Need to set the block size to 3 to get it working 
  fem::impl::assemble_cells(A.template mat_add_values<3, 3>(), g.dofmap(),
                            g.x(), cells, {dofmap.map(), 3, cells}, ident,
                            {dofmap.map(), 3, cells}, ident, {}, {}, kernel,
                            std::span<const T>(), 0, {}, {}, {});
  A.scatter_rev();

  std::vector<double> A_dense = A.to_dense();

  return A.squared_norm();
}

/// @brief Assemble P1 mass matrix and a RHS vector using element kernel
/// approaches.
///
/// Function demonstrates how hand-coded element kernels can be executed
/// in assembly over cells.
///
/// @tparam T Scalar type.
/// @param comm MPI communicator to assembler over.
template <std::floating_point T> void assemble(MPI_Comm comm) {
  // constexpr std::size_t gdim = 3;
  //  Create mesh
  auto mesh = std::make_shared<mesh::Mesh<double>>(mesh::create_box<double>(
      comm, {{{0.0, 0.0, 0.0}, {1.0, 1.0, 1.0}}}, {1, 1, 1},
      mesh::CellType::hexahedron,
      mesh::create_cell_partitioner(mesh::GhostMode::none)));

  // Create Basix P1 Lagrange element. This will be used to construct
  // basis functions inside the custom cell kernel.
  constexpr int order = 1;
  basix::FiniteElement e = basix::create_element<T>(
      basix::element::family::P,
      mesh::cell_type_to_basix_type(mesh::CellType::hexahedron), order,
      basix::element::lagrange_variant::unset,
      basix::element::dpc_variant::unset, false);

  // Construct quadrature rule
  constexpr int max_degree = 2 * order;
  auto quadrature_type = basix::quadrature::get_default_rule(
      basix::cell::type::hexahedron, max_degree);
  auto [X_b, weights] = basix::quadrature::make_quadrature<T>(
      quadrature_type, basix::cell::type::hexahedron,
      basix::polyset::type::standard, max_degree);

  mdspand_t<const T, 2> X(X_b.data(), weights.size(), 3);

  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          mesh, std::make_shared<fem::FiniteElement<double>>(
                    e, std::vector<std::size_t>{3})));

  // Build list of cells to assembler over (all cells owned by this
  // rank)
  std::int32_t size_local =
      mesh->topology()->index_map(mesh->topology()->dim())->size_local();
  std::vector<std::int32_t> cells(size_local);
  std::iota(cells.begin(), cells.end(), 0);

  // Tabulate basis functions at quadrature points
  auto e_shape = e.tabulate_shape(1, weights.size());
  std::size_t length =
      std::accumulate(e_shape.begin(), e_shape.end(), 1, std::multiplies<>{});
  std::vector<T> phi_b(length);
  mdspand_t<T, 4> phi(phi_b.data(), e_shape);

  for (int i = 0; i < 4; i++) {
    std::cout << e_shape[i] << std::endl;
  }

  e.tabulate(1, X, phi);

  // Utility function to compute det(J) for an affine triangle cell
  // (geometry is 3D)
  auto detJ = [](mdspan2_t<const T, 3, 3> x) {
    if (0)
      x(1, 0);
    return 1.0;
  };

  // Finite element mass matrix kernel function
  std::array<T, 576> A_hat_b = A_ref<T>(phi, weights);
  auto kernel_a = [A_hat = mdspan2_t<T, 24, 24>(A_hat_b.data()),
                   detJ](T *A, const T *, const T *, const T *x, const int *,
                         const uint8_t *) {
    T scale = detJ(mdspan2_t<const T, 3, 3>(x));
    mdspan2_t<T, 24, 24> _A(A);
    for (std::size_t i = 0; i < A_hat.extent(0); ++i)
      for (std::size_t j = 0; j < A_hat.extent(1); ++j)
        _A(i, j) = scale * A_hat(i, j);
  };

  //     // Assemble matrix and vector using std::function kernel
  // assemble_matrix0<T>(V, kernel_a, cells);


  //     // // Assemble matrix and vector using lambda kernel. This version
  //     // // supports efficient inlining of the kernel in the assembler. This
  //     // // can give a significant performance improvement for lightweight
  //     // // kernels.
  assemble_matrix1<T>(mesh->geometry(), *V->dofmap(), kernel_a, cells);

}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);
  assemble<double>(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
