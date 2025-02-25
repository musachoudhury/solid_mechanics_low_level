
# # %%
# A_np = np.array(A).astype(np.float64)
# b_np = np.array(b_eval).astype(np.float64)

# res = np.linalg.solve(A_np, b_np)
# res

# # %%
# # Initialize PETSc objects
# # A_petsc = PETSc.Mat().create()
# # A_petsc.setSizes(A_np.shape)
# # A_petsc.setType(PETSc.Mat.Type.DENSE)  # Set matrix type to dense
# # A_petsc.setUp()  # Finalize setup before inserting values


# def array2petsc4py(g):
#     Xpt = PETSc.Mat().createAIJ(g.shape)
#     Xpt.setUp()
#     Xpt.setValues(range(0, g.shape[0]), range(0, g.shape[1]), g)
#     Xpt.assemble()
#     return Xpt


# A_petsc = array2petsc4py(A_np)

# b_petsc = PETSc.Vec().createSeq(A_np.shape[0])
# x_petsc = PETSc.Vec().createSeq(A_np.shape[0])

# for i in range(A_np.shape[0]):
#     for j in range(A_np.shape[1]):
#         A_petsc.setValue(i, j, A_np[i, j])

# # Assemble the matrix
# A_petsc.assemble()
# # A_petsc.view()


# # Set right-hand side vector values
# for i in range(b_np.shape[0]):
#     b_petsc.setValue(i, b_np[i, 0])

# # Assemble the vector
# b_petsc.assemble()

# # Solve Ax = b using PETSc linear solver
# ksp = PETSc.KSP().create()
# ksp.setOperators(A_petsc)
# ksp.setType(PETSc.KSP.Type.PREONLY)  # Direct solve
# pc = ksp.getPC()
# pc.setType(PETSc.PC.Type.LU)  # Use LU factorization

# # Solve for x
# ksp.solve(b_petsc, x_petsc)

# # Retrieve the solution
# x_np = x_petsc.getArray()

# print("Solution x:", x_np)

# # # %%
# # A_petsc.convert("dense")
# # array = A_petsc.getDenseArray()  # Get matrix as a NumPy array
# # # # Save to a text file
# # np.savetxt("matrix.txt", array, fmt="%.6f")  # Save with 6 decimal places
# # %%

# # Create a PETSc vector

# # Save to text file
# # viewer = PETSc.Viewer().createASCII("vector_output.txt")
# # b_petsc.view(viewer)

# # res2d = res.reshape(3, 8)
# # res2d
# # # for i in range(0, 8):
# # #     for j in range(0, 8):
# # #         calc_B(phi[i], x_1, x_2, x_3)*
# # N_np = np.array(N.evalf(subs={x_1: 0.5, x_2: 0.5, x_3: 0.5}, n=15)).astype(np.float64)
# # for i in range(0, 8):
# #     # u at QP
# #     u_qp = np.matmul(N_np, res)

# #     # strain = B *u_

# #     B = calc_B(phi[i], x_1, x_2, x_3)
# #     B_np = np.array(B.evalf(subs={x_1: 0.5, x_2: 0.5, x_3: 0.5}, n=15)).astype(
# #         np.float64
# #     )

# # strain = np.matmul(B_np, res2d)
# # %%