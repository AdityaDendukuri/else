# Block-tridiagonal solver

ELSE repeatedly solves

\[
(-R)X=B,
\]

where `R` is the truncated generator of the current subnetwork. For a reaction
network, the states can often be divided into levels such that reactions stay
within one level or move to an adjacent level. Ordering states by these levels
makes \(-R\) block tridiagonal, so it can be factored by the block analogue of
the Thomas algorithm.

## 1. Supplying the levels

ELSE does not infer levels from a generator or from a reaction system. The
caller supplies a function \(\ell(x)\) assigning a level label to each state.
The labels are compressed to \(0,\ldots,p-1\) before the matrix is assembled
into blocks.

For example, a CME caller may choose a coordinate \(q\) for which

\[
h_q=\max_r |(\nu_r)_q|>0.
\]

and supply

\[
\ell(x)=\left\lfloor\frac{x_q}{h_q}\right\rfloor.
\]

Since every reaction satisfies

\[
|(\nu_r)_q|\le h_q,
\]

every generator entry connects states whose levels differ by at most one:

\[
R_{yx}\ne 0 \quad\Longrightarrow\quad
|\ell(y)-\ell(x)|\le 1.
\]

If the selected coordinate is constant on the current subnetwork, there is
only one level and `Subnetwork` uses its dense LU path. The block solver
also checks every supplied matrix entry and rejects an ordering that is not
block tridiagonal.

Other callers can supply levels obtained from their own structure. For an
arbitrary graph, for example, connected components can be separated first and
BFS distance can provide levels within each component.

## 2. Block form

After permuting states so that equal levels are contiguous, write

\[
A=-R=
\begin{bmatrix}
D_0 & U_0 \\
L_0 & D_1 & U_1 \\
    & L_1 & D_2 & \ddots \\
    &     & \ddots & \ddots
\end{bmatrix}.
\]

The block sizes need not be equal. If level \(k\) contains \(n_k\) states,
then

\[
D_k\in\mathbb R^{n_k\times n_k},\qquad
U_k\in\mathbb R^{n_k\times n_{k+1}},\qquad
L_k\in\mathbb R^{n_{k+1}\times n_k}.
\]

The implementation records both permutations:

- `order[new] = old`, used to gather and scatter right-hand sides;
- `inverse_order[old] = new`, used to assemble the blocks from CSR entries.

Here, *scatter* only means copying the solved entries from level order back to
the original state order.

## 3. Factorization

Define the successive Schur complements

\[
S_0=D_0,
\]

and, for \(k=1,\ldots,p-1\),

\[
M_k=L_{k-1}S_{k-1}^{-1},
\qquad
S_k=D_k-M_kU_{k-1}.
\]

The code never forms \(S_{k-1}^{-1}\). It computes each row of \(M_k\) by a
transpose solve:

\[
S_{k-1}^{T}(M_k)_{i,:}^{T}=(L_{k-1})_{i,:}^{T}.
\]

It then stores \(M_k\) in place of the original lower block and stores an
unpivoted dense LU factorization of every \(S_k\). Thus

\[
A=
\begin{bmatrix}
I \\
M_1&I \\
&M_2&I\\
&&\ddots&\ddots
\end{bmatrix}
\begin{bmatrix}
S_0&U_0\\
&S_1&U_1\\
&&S_2&\ddots\\
&&&\ddots
\end{bmatrix}.
\]

For an eventually escaping finite subnetwork, \(-R\) is a nonsingular
M-matrix. Its principal Schur complements retain the M-matrix structure, so
the exact-arithmetic elimination has nonzero positive pivots without row
pivoting. The implementation therefore preserves the block structure rather
than applying general sparse pivoting. This structural result does not make
finite-precision failure impossible; a singular or numerically invalid Schur
block is reported as an error.

## 4. Solving

Partition the right-hand side into the same levels,

\[
B=(B_0^T,\ldots,B_{p-1}^T)^T.
\]

Forward substitution through the first block factor gives

\[
Y_0=B_0,
\qquad
Y_k=B_k-M_kY_{k-1}.
\]

Backward substitution through the second factor gives

\[
X_{p-1}=S_{p-1}^{-1}Y_{p-1},
\]

\[
X_k=S_k^{-1}(Y_k-U_kX_{k+1}),
\qquad k=p-2,\ldots,0.
\]

Finally, the solution is copied from level order back to the original state
order. The matrix-right-hand-side implementation performs these operations
for all entrance states together. The vector solve is simply the same routine
with one column.

## 5. Symmetric positive-definite specialization

For a reversible generator after stationary-weight symmetrization, and for a
restricted Laplacian, the block-tridiagonal matrix is symmetric positive
definite. Write its lower off-diagonal blocks as \(E_k\). Block Cholesky uses

\[
G_0G_0^T=D_0,
\qquad
C_k=E_{k-1}G_{k-1}^{-T},
\qquad
G_kG_k^T=D_k-C_kC_k^T.
\]

Only the lower multipliers \(C_k\) and diagonal Cholesky factors \(G_k\) are
stored. Forward and backward substitution are

\[
G_kY_k=B_k-C_kY_{k-1},
\qquad
G_k^TX_k=Y_k-C_{k+1}^TX_{k+1}.
\]

This path avoids the separate upper blocks and LU transpose solves. General
nonsymmetric generators continue to use block LU, and complex Talbot-shifted
systems are not treated as SPD.

## 6. Cost

For block sizes \(n_0,\ldots,n_{p-1}\), factorization costs approximately

\[
O\!\left(
\sum_k n_k^3
+\sum_{k=1}^{p-1}
  (n_kn_{k-1}^2+n_k^2n_{k-1})
\right).
\]

Solving for \(s\) right-hand sides costs

\[
O\!\left(
s\sum_k n_k^2
+s\sum_{k=1}^{p-1}n_kn_{k-1}
\right).
\]

When every block has size one, these reduce to the usual linear-time scalar
Thomas algorithm. When one level contains nearly every state, the method
approaches dense LU and provides little benefit.

The diagonal Schur blocks and lower multipliers are stored densely because
elimination generally fills them in even when the original generator is
sparse. The implementation also stores the adjacent upper blocks densely.
For the small ELSE capacities this keeps the algorithm short and gives
contiguous arithmetic; a sparse upper-block representation would save some
zeros but would add another matrix format and another multiplication kernel.

## 7. Use in ELSE

The trajectory and density builders follow the same sequence:

1. Receive a level function from the caller.
2. Enumerate the states of a subnetwork.
3. Evaluate and compress the supplied level labels.
4. Assemble the truncated generator in CSR form.
5. Pass the explicit levels to `Subnetwork`.
6. Factor \(-R\) once and reuse it for occupation and moment solves.

The solver is therefore optional and explicit. It does not claim that an
arbitrary sparse matrix is block tridiagonal, and it has no graph-search
fallback. Reversible subnetworks continue to use the stationary-weighted
Cholesky path instead.

The implementation and label normalization are in
[`include/else/block_tridiagonal.hpp`](../include/else/block_tridiagonal.hpp).
