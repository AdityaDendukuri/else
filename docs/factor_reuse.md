# Reusing factorizations between ELSE steps

## 1. Setting

Let

\[
A_0=-R_0
\]

be the matrix from the last ELSE step that was factorized, and let

\[
A=-R=A_0+\Delta
\]

be the matrix for the current subnetwork. Both matrices have order \(n\). In a
CME subnetwork, replacing one state changes its diagonal, its outgoing
transition column, and the rows receiving its incoming transitions. Thus a
state replacement changes a row and a column, even when the generator is
sparse.

The method retains the factorization of \(A_0\) and represents \(\Delta\) as a
small-rank row-and-column correction. It does **not** form \(A_0^{-1}\).
Applications of \(A_0^{-1}\) are triangular solves with the retained dense LU,
dense Cholesky, block LU, or block Cholesky factors.

## 2. Persistent state IDs and stable matrix slots

Each state receives an immutable graph ID the first time it is discovered.
The active subnetwork maps these graph IDs to matrix slots. When shedding
removes an old state, a newly admitted state fills the vacated slot; all other
states retain their row and column numbers.

Suppose the slot set

\[
S=\{s_1,\ldots,s_q\}
\]

contains every slot whose state differs between \(A_0\) and \(A\). Stable slots
then imply

\[
\Delta_{ij}=0, \qquad i\notin S,\quad j\notin S.
\]

Therefore every nonzero of \(\Delta\) lies in a changed row or changed column.
Without stable slots, an arbitrary reordering could make unchanged states
appear to change and destroy this low-rank structure.

The discovered-state graph belongs to one simulation rather than global
static storage. This retains stable IDs without coupling different reaction
systems, simulations, or threads.

## 3. Row-and-column decomposition

Define the selection matrix

\[
E=\begin{bmatrix}e_{s_1}&\cdots&e_{s_q}\end{bmatrix}\in\mathbb R^{n\times q},
\]

where \(e_j\) is the \(j\)-th coordinate vector. The changed rows are

\[
D_r=E^T\Delta\in\mathbb R^{q\times n}.
\]

After the changed rows have been included, the remaining part of the changed
columns is

\[
D_c=(I-EE^T)\Delta E\in\mathbb R^{n\times q}.
\]

The factor \(I-EE^T\) zeros the rows in \(S\), preventing the
\(S\)-by-\(S\) block from being counted twice. Entrywise,

\[
\begin{aligned}
\Delta
&=EE^T\Delta+(I-EE^T)\Delta EE^T\\
&=E D_r+D_c E^T.
\end{aligned}
\]

Set

\[
U=\begin{bmatrix}E&D_c\end{bmatrix},
\qquad
V^T=\begin{bmatrix}D_r\\E^T\end{bmatrix}.
\]

Then

\[
\boxed{A=A_0+UV^T},
\qquad
U,V\in\mathbb R^{n\times r},
\qquad
r\le 2q.
\]

The implementation constructs these factors directly from the sparse rows of
\(R_0\) and \(R\). It does not construct a dense \(n\)-by-\(n\) difference.

## 4. Reusing the stored factorization

Let

\[
Z_0=A_0^{-1}.
\]

First solve with the stored factors for every column of \(U\):

\[
W=Z_0U.
\]

This requires \(r\le 2q\) solves with the existing large factorization. Form
the compact matrix

\[
K=I_r+V^TW.
\]

The Woodbury identity gives

\[
\begin{aligned}
A^{-1}
&=(A_0+UV^T)^{-1}\\
&=Z_0-Z_0U(I_r+V^TZ_0U)^{-1}V^TZ_0\\
&=Z_0-WK^{-1}V^TZ_0.
\end{aligned}
\]

Hence a solve \(Ax=b\) is evaluated as

\[
\begin{aligned}
y&=Z_0b,\\
c&=K^{-1}V^Ty,\\
x&=y-Wc.
\end{aligned}
\]

For multiple right-hand sides \(B\), the same correction is reused:

\[
Y=Z_0B,
\qquad
C=K^{-1}V^TY,
\qquad
X=Y-WC.
\]

The current code factorizes only \(K\), whose order is at most \(2q\). The
large \(n\)-by-\(n\) factorization of \(A_0\) is retained. Although \(A_0\) may
be symmetric positive definite and solved by Cholesky, the general
row-and-column representation can produce a nonsymmetric \(K\), so the small
matrix currently uses unpivoted LU.

If \(A_0\) and \(A\) are nonsingular, this is an algebraic identity rather than
an approximation. Numerical differences arise only from floating-point
factorization and triangular solves.

## 5. ELSE occupation solves

For entrance matrix \(P\), ELSE requires

\[
X=A^{-1}P,
\qquad
Y=A^{-1}X=A^{-2}P.
\]

The first matrix supplies occupation times and exit probabilities. The second
supplies the time-weighted occupation used for conditional mean exit times.
Both solves use the same \(W\) and factorization of \(K\):

\[
\begin{aligned}
X&=Z_0P-WK^{-1}V^TZ_0P,\\
Y&=Z_0X-WK^{-1}V^TZ_0X.
\end{aligned}
\]

Thus the correction is constructed once per macrostep and reused for every
entrance column and for both occupation solves.

## 6. Cost

Let \(T_s(n)\) be the cost of one solve with the stored factors and let
\(r\le2q\). Correction construction costs approximately

\[
rT_s(n)+O(nr^2)+O(r^3).
\]

Each corrected right-hand side then costs

\[
T_s(n)+O(nr+r^2).
\]

For dense factors, \(T_s(n)=O(n^2)\), so construction costs

\[
O(rn^2+nr^2+r^3),
\]

compared with \(O(n^3)\) for dense refactorization. Reuse is therefore most
effective when \(q\ll n\).

For the block-tridiagonal solver, both factorization and solves are already
much cheaper than their dense counterparts. Several additional block solves
can consequently cost as much as a fresh block factorization. The percentage
of unchanged matrix entries alone does not determine the crossover.

## 7. Rebasing and numerical fallback

The stored factors remain factors of \(A_0\); Woodbury supplies the action of
\(A^{-1}\), but it does not turn the old factors into factors of \(A\). If the
base were retained indefinitely, the number of slots differing from it would
grow and so would the correction rank.

The current trajectory implementation therefore uses a short-update policy:

1. Reuse the stored factors while at most three slots differ from the base.
2. If more than three slots differ, factorize the current matrix normally.
3. Make that current matrix and its factors the new base.

The value three is an empirical guard for the fast block-tridiagonal backend,
not a mathematical limit. The dense results below show that a larger cutoff is
appropriate for dense LU or Cholesky.

For an updated solve, the code checks

\[
\frac{\lVert Ax-b\rVert_\infty}
     {\max(1,\lVert b\rVert_\infty)}\le10^{-8}.
\]

A singular compact factorization or failed residual check triggers a complete
factorization of the current matrix and establishes a new base.

## 8. Dense factor benchmark

The benchmark replaced rows and columns of a dense \(600\)-by-\(600\) positive
definite matrix. “Update” includes construction of \(U\), \(V\), \(W\), and
factorization of \(K\). “Rebuild” factorizes the complete current matrix. Each
row is the mean of eight repetitions.

| Changed states \(q\) | Rank bound \(2q\) | LU rebuild (ms) | LU update (ms) | LU speedup | Cholesky rebuild (ms) | Cholesky update (ms) | Cholesky speedup |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1  | 2   | 16.059 | 0.716  | 22.43 | 23.294 | 0.865  | 26.94 |
| 2  | 4   | 16.066 | 1.433  | 11.21 | 23.114 | 1.730  | 13.36 |
| 4  | 8   | 16.098 | 2.890  | 5.57  | 23.109 | 3.477  | 6.65  |
| 8  | 16  | 16.095 | 5.902  | 2.73  | 23.149 | 7.093  | 3.26  |
| 12 | 24  | 16.118 | 9.089  | 1.77  | 23.495 | 10.758 | 2.18  |
| 16 | 32  | 16.308 | 12.244 | 1.33  | 23.247 | 14.817 | 1.57  |
| 24 | 48  | 16.100 | 19.072 | 0.84  | 23.254 | 22.448 | 1.04  |
| 32 | 64  | 15.958 | 25.977 | 0.61  | 23.408 | 30.472 | 0.77  |

The LU crossover lies between 16 and 24 replaced states. The Cholesky
crossover is near 24 states. The maximum difference between a corrected solve
and a solve using freshly computed factors was approximately
\(5\times10^{-15}\).

Reproduce with

```sh
./build/benchmarks/dense_factor_update 600
```

## 9. Oregonator benchmark

The Oregonator benchmark used one trajectory, a subnetwork capacity of 600,
expansion depth one, and 1000 ELSE macrosteps. “Dense” assigns every state to
one block and therefore uses a single dense LU block. “Block” uses total copy
number divided by two as the block level.

| Backend | Scratch factorization (s) | Factor reuse (s) | Speedup |
|---|---:|---:|---:|
| Dense LU | 15.47 | 3.15 | 4.90 |
| Block-tridiagonal LU | 0.93 | 0.61 | 1.54 |

Dense reuse is effective, but block-tridiagonal reuse remains about five times
faster in absolute wall time. The smaller block speedup is expected: the block
factorization being avoided is already inexpensive.

Oregonator is nonsymmetric and therefore cannot use Cholesky directly. The
Cholesky results in the preceding table apply to reversible or otherwise SPD
systems.

Reproduce with

```sh
./build/benchmarks/factor_reuse_cme 1000 1 600 dense
./build/benchmarks/factor_reuse_cme 1000 1 600 block
```

## 10. Relation to direct block-factor reuse

The current implementation uses the block LU or block Cholesky factors only
through applications of \(A_0^{-1}\) inside Woodbury. It does not update the
block-Thomas Schur factors themselves.

For a block-tridiagonal matrix with diagonal blocks \(D_k\), lower blocks
\(L_k\), and upper blocks \(B_k\), block Thomas forms

\[
S_0=D_0,
\qquad
S_k=D_k-L_{k-1}S_{k-1}^{-1}B_{k-1}.
\]

If the first changed block is \(j\), the factors of
\(S_0,\ldots,S_{j-1}\) can be retained and only the suffix beginning at
\(S_j\) must be recomputed. Such a direct block-factor update would eliminate
the Woodbury solves and make the new factors the base for the following step.
It is a distinct method and has not yet replaced the current Woodbury path.
