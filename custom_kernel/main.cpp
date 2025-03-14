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
std::array<T, 3 * 3 * 8 * 8> A_ref(mdspand_t<const T, 4> phi,
                                   std::span<const T> w) {
  std::array<T, 9> A_b{};
  mdspan2_t<T, 3, 3> A(A_b.data());

  std::array<T, 3 * 3 * 8 * 8> AFull_b{};
  // for (std::size_t i = 0; i < 576; ++i) {
  //   AFull_b[i] = 1.0;
  // }
  mdspan2_t<T, 3 * 8, 3 * 8> AFull(AFull_b.data());

  // A.extent(0) = 24
  // A.extent(1) = 24
  for (std::size_t k = 0; k < phi.extent(1); ++k) {       // quadrature point k
    for (std::size_t i = 0; i < AFull.extent(0) / 3; ++i) // row i
    {
      for (std::size_t j = 0; j < AFull.extent(1) / 3; ++j) // column j
      {
        std::array<T, 6 * 6> tangent_b{};
        mdspan2_t<T, 6, 6> tangent(tangent_b.data());

        double E = 200000.0;
        double nu = 0.3;

        double lmbda = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu));
        double mu = E / (2.0 * (1.0 + nu));

        for (std::size_t p = 0; p < 3; p++) {
          for (std::size_t q = 0; q < 3; q++) {
            tangent(p, q) = lmbda;
          }
          tangent(p, p) = lmbda + 2.0 * mu;
        }

        for (std::size_t p = 3; p < 6; p++)
          tangent(p, p) = mu;

        std::array<T, 6 * 3> B_T_b{};
        mdspan2_t<T, 3, 6> B_T(B_T_b.data());

        std::array<T, 6 * 3> B_b{};
        mdspan2_t<T, 6, 3> B(B_b.data());

        // Move out of loop
        B_T(0, 0) = phi(basix::indexing::idx(1, 0, 0), k, i, 0);
        B_T(1, 1) = phi(basix::indexing::idx(0, 1, 0), k, i, 0);
        B_T(2, 2) = phi(basix::indexing::idx(0, 0, 1), k, i, 0);

        B_T(1, 3) = phi(basix::indexing::idx(0, 0, 1), k, i, 0);
        B_T(2, 3) = phi(basix::indexing::idx(0, 1, 0), k, i, 0);

        B_T(0, 4) = phi(basix::indexing::idx(0, 0, 1), k, i, 0);
        B_T(2, 4) = phi(basix::indexing::idx(1, 0, 0), k, i, 0);

        B_T(0, 5) = phi(basix::indexing::idx(0, 1, 0), k, i, 0);
        B_T(1, 5) = phi(basix::indexing::idx(1, 0, 0), k, i, 0);

        ////////////////////////////////////////////////////////////////////////////

        B(0, 0) = phi(basix::indexing::idx(1, 0, 0), k, j, 0);
        B(1, 1) = phi(basix::indexing::idx(0, 1, 0), k, j, 0);
        B(2, 2) = phi(basix::indexing::idx(0, 0, 1), k, j, 0);

        B(3, 1) = phi(basix::indexing::idx(0, 0, 1), k, j, 0);
        B(3, 2) = phi(basix::indexing::idx(0, 1, 0), k, j, 0);

        B(4, 0) = phi(basix::indexing::idx(0, 0, 1), k, j, 0);
        B(4, 2) = phi(basix::indexing::idx(1, 0, 0), k, j, 0);

        B(5, 0) = phi(basix::indexing::idx(0, 1, 0), k, j, 0);
        B(5, 1) = phi(basix::indexing::idx(1, 0, 0), k, j, 0);

        // Write B.T * D * B here

        // C = B.T * D    3x6x6x6 = 3x6

        std::array<T, 6 * 3> C_b{};
        mdspan2_t<T, 3, 6> C(C_b.data());

        for (std::size_t p = 0; p < 3; ++p)
          for (std::size_t q = 0; q < 6; ++q)
            for (std::size_t r = 0; r < 6; ++r)
              C(p, q) += B_T(p, r) * tangent(r, q);

        // A = C * B    3x6x6x3 = 3x3

        for (std::size_t p = 0; p < 3; ++p) {
          for (std::size_t q = 0; q < 3; ++q) {
            for (std::size_t r = 0; r < 6; ++r) {
              AFull(3 * i + p, 3 * j + q) += -w[k] * C(p, r) * B(r, q);
            }
            // std::cout << AFull(3 * i + p, 3 * j + q) << ", ";
            // std::cout << 3 * i + p << ", " << 3 * j + q << ", ";
          }
          // std::cout << "\n";
        }

        // for (std::size_t p = 0; p < 3; ++p)
        //   for (std::size_t q = 0; q < 3; ++q) {
        //     AFull(3 * i + p, 3 * j + q) = A(p, q);
        //   }
      }
    }
  }

  return AFull_b;
}

/// @brief Compute the P1 RHS vector for f=1 on the reference cell.
/// @tparam T Scalar type.
/// @param phi Basis functions.
/// @param w Integration weights.
/// @return RHS reference vector.
template <typename T>
std::array<T, 3 * 8> b_ref(mdspand_t<const T, 4> phi, std::span<const T> w) {

  std::array<T, 3 * 8> b;

  std::array<T, 3 * 3> N_T_b;
  mdspan2_t<T, 3, 3> N_T(N_T_b.data());

  std::array<T, 3 * 6> B_T_b;
  mdspan2_t<T, 3, 6> B_T(B_T_b.data());

  std::array<T, 6> stress_b;
  mdspan2_t<T, 6, 1> stress(stress_b.data());

  std::array<T, 3> body_force_b{1000, 0, 0};
  mdspan2_t<T, 3, 1> body_force(body_force_b.data());

  for (std::size_t k = 0; k < phi.extent(1); ++k)  // quadrature point
    for (std::size_t i = 0; i < b.size() / 3; ++i) // row i
    {

      for (std::size_t j = 0; j < 3; j++) {
        N_T(j, j) = phi(0, k, i, 0);
      }

      B_T(0, 0) = phi(basix::indexing::idx(1, 0, 0), k, i, 0);
      B_T(1, 1) = phi(basix::indexing::idx(0, 1, 0), k, i, 0);
      B_T(2, 2) = phi(basix::indexing::idx(0, 0, 1), k, i, 0);

      B_T(1, 3) = phi(basix::indexing::idx(0, 0, 1), k, i, 0);
      B_T(2, 3) = phi(basix::indexing::idx(0, 1, 0), k, i, 0);

      B_T(0, 4) = phi(basix::indexing::idx(0, 0, 1), k, i, 0);
      B_T(2, 4) = phi(basix::indexing::idx(1, 0, 0), k, i, 0);

      B_T(0, 5) = phi(basix::indexing::idx(0, 1, 0), k, i, 0);
      B_T(1, 5) = phi(basix::indexing::idx(1, 0, 0), k, i, 0);

      for (std::size_t j = 0; j < 3; j++) {

        for (std::size_t p = 0; p < 6; p++) {
          b[3 * i + j] += w[k] * (B_T(j, p) * stress(p, 1));
        }
        // Maybe we can more efficient because N_T is diagonal
        // for (std::size_t p = 0; p < 3; p++) {
        //   b[3 * i + j] += w[k] * (N_T(j, p) * body_force(p, 1));
        //   std::cout << 3 * i + j << std::endl; 
        // }
      }
    }

  return b;
}

/// @brief Assemble a matrix operator using a `std::function` kernel
/// function.
/// @tparam T Scalar type.
/// @param V Function space.
/// @param kernel Element kernel to execute.
/// @param cells Cells to execute the kernel over.
/// @return Frobenius norm squared of the matrix.
template <std::floating_point T>
double assemble_matrix0(std::shared_ptr<fem::FunctionSpace<T>> V, auto kernel,
                        std::span<const std::int32_t> cells) {
  // Kernel data (ID, kernel function, cell indices to execute over)
  std::vector kernel_data{
      fem::integral_data<T>(-1, kernel, cells, std::vector<int>{})};

  // Associate kernel with cells (as opposed to facets, etc)
  std::map integrals{std::pair{fem::IntegralType::cell, kernel_data}};

  fem::Form<T> a({V, V}, integrals, {}, {}, false, {}, V->mesh());
  auto dofmap = V->dofmap();
  auto sp = la::SparsityPattern(
      V->mesh()->comm(), {dofmap->index_map, dofmap->index_map},
      {dofmap->index_map_bs(), dofmap->index_map_bs()});
  fem::sparsitybuild::cells(sp, {cells, cells}, {*dofmap, *dofmap});
  sp.finalize();
  la::MatrixCSR<T> A(sp);
  common::Timer timer("Assembler0 std::function (matrix)");
  assemble_matrix(A.mat_add_values(), a, {});
  A.scatter_rev();
  return A.squared_norm();
}

/// @brief Assemble a RHS vector using a `std::function` kernel
/// function.
/// @tparam T Scalar type.
/// @param V Function space.
/// @param kernel Element kernel to execute.
/// @param cells Cells to execute the kernel over.
/// @return l2 norm squared of the vector.
template <std::floating_point T>
double assemble_vector0(std::shared_ptr<fem::FunctionSpace<T>> V, auto kernel,
                        std::span<const std::int32_t> cells) {
  auto mesh = V->mesh();
  std::vector kernal_data{
      fem::integral_data<T>(-1, kernel, cells, std::vector<int>{})};
  std::map integrals{std::pair{fem::IntegralType::cell, kernal_data}};
  fem::Form<T> L({V}, integrals, {}, {}, false, {}, mesh);
  auto dofmap = V->dofmap();
  la::Vector<T> b(dofmap->index_map, 1);
  common::Timer timer("Assembler0 std::function (vector)");
  fem::assemble_vector(b.mutable_array(), L);
  b.scatter_rev(std::plus<T>());
  return la::squared_norm(b);
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
  fem::impl::assemble_cells(A.template mat_add_values<3, 3>(), g.dofmap(),
                            g.x(), cells, {dofmap.map(), 3, cells}, ident,
                            {dofmap.map(), 3, cells}, ident, {}, {}, kernel,
                            std::span<const T>(), 0, {}, {}, {});
  A.scatter_rev();

  // std::vector<double> A_dense = A.to_dense();

  // for (int i = 0; i < 3*8; i++) {
  //   for (int j = 0; j < 3*8; j++) {
  //     std::cout << A_dense[i * 3*8 + j] << ", ";
  //   }
  //   std::cout << "\n";
  // }

  return A.squared_norm();
}

/// @brief Assemble a RHS vector using using a lambda kernel function.
///
/// The lambda function can be inlined in the assembly code, which can
/// be important for performance for lightweight kernels.
///
/// @tparam T Scalar type.
/// @param g mesh geometry.
/// @param dofmap dofmap.
/// @param kernel Element kernel to execute.
/// @param cells Cells to execute the kernel over.
/// @return l2 norm squared of the vector.
template <std::floating_point T>
double assemble_vector1(const mesh::Geometry<T> &g, const fem::DofMap &dofmap,
                        auto kernel, const std::vector<std::int32_t> &cells) {
  la::Vector<T> b(dofmap.index_map, 1);
  common::Timer timer("Assembler1 lambda (vector)");
  fem::impl::assemble_cells<T, 1>(
      [](auto, auto, auto, auto) {}, b.mutable_array(), g.dofmap(), g.x(),
      cells, {dofmap.map(), 1, cells}, kernel, {}, {}, 0, {});
  b.scatter_rev(std::plus<T>());

    std::span<T> b_span = b.mutable_array();
  for (int i=0;i<24;i++){
    std::cout << b_span[i] << std::endl;
  }
  return la::squared_norm(b);
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
  constexpr int order = 2;
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

  // Create a scalar function space
  // auto V =
  // std::make_shared<fem::FunctionSpace<T>>(fem::create_functionspace<T>(
  //     mesh, std::make_shared<fem::FiniteElement<T>>(
  //               e, std::vector<std::size_t>{gdim})));

  auto V = std::make_shared<fem::FunctionSpace<double>>(
      fem::create_functionspace<double>(
          mesh, std::make_shared<fem::FiniteElement<double>>(
                    e, std::vector<std::size_t>{3})));

  auto dofmap = V->dofmap(); //->map();

  auto dmap = dofmap->map();

  const int num_dofs = dmap.extent(1);
  std::cout << num_dofs << std::endl;

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
  // std::cout << "Phi shape 1: " << phi_b.data() << "shape 2: " <<
  // e_shape <<
  // "\n";
  e.tabulate(1, X, phi);

  // Utility function to compute det(J) for an affine triangle cell
  //(geometry is 3D)
  auto detJ = [](mdspan2_t<const T, 3, 3> x) {
    if (0)
      x(1, 0);
    return 1.0; // std::abs((x(0, 0) - x(1, 0)) * (x(2, 1) - x(1, 1)) -
                //      (x(0, 1) - x(1, 1)) * (x(2, 0) - x(1, 0)));
  };

  // Finite element mass matrix kernel function
  std::array<T, 3 * 3 * 8 * 8> A_hat_b = A_ref<T>(phi, weights);

  mdspan2_t<T, 3 * 8, 3 * 8> A_hat(A_hat_b.data());

  for (std::size_t i = 0; i < A_hat.extent(0); i++) {
    for (std::size_t j = 0; j < A_hat.extent(1); j++)
      std::cout << A_hat(i, j) << ", ";
    std::cout << "\n";
  }
  auto kernel_a = [A_hat = mdspan2_t<T, 3 * 8, 3 * 8>(A_hat_b.data()),
                   detJ](T *A, const T *, const T *, const T *x, const int *,
                         const uint8_t *) {
    T scale = detJ(mdspan2_t<const T, 3, 3>(x));
    mdspan2_t<T, 3 * 8, 3 * 8> _A(A);
    for (std::size_t i = 0; i < A_hat.extent(0); ++i)
      for (std::size_t j = 0; j < A_hat.extent(1); ++j)
        _A(i, j) = scale * A_hat(i, j);
  };

  // Finite element RHS (f=1) kernel function
  auto kernel_L = [b_hat = b_ref<T>(phi, weights),
                   detJ](T *b, const T *, const T *, const T *x, const int *,
                         const uint8_t *) {
    T scale = detJ(mdspan2_t<const T, 3, 3>(x));
    for (std::size_t i = 0; i < 3; ++i)
      b[i] = scale * b_hat[i];
  };

  //     // Assemble matrix and vector using std::function kernel
  // assemble_matrix0<T>(V, kernel_a, cells);
  //     //assemble_vector0<T>(V, kernel_L, cells);

  //     // // Assemble matrix and vector using lambda kernel. This version
  //     // // supports efficient inlining of the kernel in the assembler.
  //     This
  //     // // can give a significant performance improvement for
  //     lightweight
  //     // // kernels.
  assemble_matrix1<T>(mesh->geometry(), *V->dofmap(), kernel_a, cells);
  assemble_vector1<T>(mesh->geometry(), *V->dofmap(), kernel_L, cells);

  //     //const fem::DofMap dofmap = *V->dofmap().map();

  //     //const fem::DofMap &dofmap =
  //     auto dofmap = V->dofmap();//->map();

  //     auto dmap = dofmap->map();

  //     const int num_dofs = dmap.extent(1);
  //     std::cout << num_dofs << std::endl;
  // list_timings(comm);
}

int main(int argc, char *argv[]) {
  MPI_Init(&argc, &argv);
  dolfinx::init_logging(argc, argv);
  assemble<double>(MPI_COMM_WORLD);
  // assemble<double>(MPI_COMM_WORLD);
  MPI_Finalize();
  return 0;
}
