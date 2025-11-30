"""
Matrix Operations
=================

Matrix operations over a finite field.
"""

from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from .field import FiniteField


class Matrix:
    """Matrix operations over a finite field."""

    def __init__(self, data: List[List[int]], field: 'FiniteField'):
        """
        Initialize a matrix with the given data over a finite field.

        Args:
            data: 2D list of integers representing matrix elements
            field: The finite field for operations
        """
        self.data = data
        self.field = field
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0

    @classmethod
    def identity(cls, n: int, field: 'FiniteField') -> 'Matrix':
        """Create an n×n identity matrix."""
        data = [[1 if i == j else 0 for j in range(n)] for i in range(n)]
        return cls(data, field)

    @classmethod
    def random(cls, rows: int, cols: int, field: 'FiniteField') -> 'Matrix':
        """Create a random matrix."""
        data = [[field.random() for _ in range(cols)] for _ in range(rows)]
        return cls(data, field)

    @classmethod
    def random_invertible(cls, n: int, field: 'FiniteField') -> 'Matrix':
        """
        Create a random invertible matrix using LU decomposition approach.

        Generates L (lower triangular with 1s on diagonal) × U (upper triangular
        with non-zero diagonal), which is guaranteed invertible.

        Args:
            n: Matrix dimension
            field: The finite field for operations

        Returns:
            A random invertible n×n matrix
        """
        # Create lower triangular matrix with 1s on diagonal
        L_data = [[0] * n for _ in range(n)]
        for i in range(n):
            L_data[i][i] = 1
            for j in range(i):
                L_data[i][j] = field.random()

        # Create upper triangular matrix with non-zero diagonal
        U_data = [[0] * n for _ in range(n)]
        for i in range(n):
            U_data[i][i] = field.random_nonzero()
            for j in range(i + 1, n):
                U_data[i][j] = field.random()

        L = cls(L_data, field)
        U = cls(U_data, field)

        return L.multiply(U)

    @classmethod
    def diagonal(cls, diag: List[int], field: 'FiniteField') -> 'Matrix':
        """Create a diagonal matrix from a list of diagonal elements."""
        n = len(diag)
        data = [[diag[i] if i == j else 0 for j in range(n)] for i in range(n)]
        return cls(data, field)

    def multiply(self, other: 'Matrix') -> 'Matrix':
        """
        Matrix multiplication.

        Args:
            other: Matrix to multiply with

        Returns:
            Product matrix self × other

        Raises:
            ValueError: If dimensions are incompatible
        """
        if self.cols != other.rows:
            raise ValueError("Matrix dimensions incompatible for multiplication")

        result = [[0] * other.cols for _ in range(self.rows)]
        for i in range(self.rows):
            for j in range(other.cols):
                s = 0
                for k in range(self.cols):
                    s = self.field.add(s, self.field.mul(self.data[i][k], other.data[k][j]))
                result[i][j] = s

        return Matrix(result, self.field)

    def multiply_vector(self, vec: List[int]) -> List[int]:
        """
        Multiply matrix by a column vector.

        Args:
            vec: Column vector as a list

        Returns:
            Result vector

        Raises:
            ValueError: If dimensions are incompatible
        """
        if len(vec) != self.cols:
            raise ValueError("Vector dimension incompatible with matrix")

        result = []
        for i in range(self.rows):
            s = 0
            for j in range(self.cols):
                s = self.field.add(s, self.field.mul(self.data[i][j], vec[j]))
            result.append(s)
        return result

    def add_matrix(self, other: 'Matrix') -> 'Matrix':
        """
        Matrix addition.

        Args:
            other: Matrix to add

        Returns:
            Sum matrix

        Raises:
            ValueError: If dimensions don't match
        """
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match for addition")

        result = [[self.field.add(self.data[i][j], other.data[i][j])
                   for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result, self.field)

    def scalar_multiply(self, scalar: int) -> 'Matrix':
        """Multiply matrix by a scalar."""
        result = [[self.field.mul(self.data[i][j], scalar)
                   for j in range(self.cols)] for i in range(self.rows)]
        return Matrix(result, self.field)

    def transpose(self) -> 'Matrix':
        """Compute matrix transpose."""
        result = [[self.data[j][i] for j in range(self.rows)] for i in range(self.cols)]
        return Matrix(result, self.field)

    def inverse(self) -> 'Matrix':
        """
        Compute matrix inverse using Gaussian elimination.

        Returns:
            Inverse matrix

        Raises:
            ValueError: If matrix is not square or not invertible
        """
        if self.rows != self.cols:
            raise ValueError("Only square matrices can be inverted")

        n = self.rows
        # Create augmented matrix [A | I]
        aug = [[self.data[i][j] for j in range(n)] +
               [1 if i == k else 0 for k in range(n)]
               for i in range(n)]

        # Forward elimination
        for col in range(n):
            # Find pivot
            pivot_row = None
            for row in range(col, n):
                if aug[row][col] != 0:
                    pivot_row = row
                    break

            if pivot_row is None:
                raise ValueError("Matrix is not invertible")

            # Swap rows
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

            # Scale pivot row
            pivot_inv = self.field.inv(aug[col][col])
            for j in range(2 * n):
                aug[col][j] = self.field.mul(aug[col][j], pivot_inv)

            # Eliminate column
            for row in range(n):
                if row != col and aug[row][col] != 0:
                    factor = aug[row][col]
                    for j in range(2 * n):
                        aug[row][j] = self.field.sub(
                            aug[row][j],
                            self.field.mul(factor, aug[col][j])
                        )

        # Extract inverse
        inv_data = [[aug[i][j + n] for j in range(n)] for i in range(n)]
        return Matrix(inv_data, self.field)

    def to_list(self) -> List[List[int]]:
        """Convert matrix to a 2D list."""
        return self.data

    def __repr__(self) -> str:
        return f"Matrix({self.rows}x{self.cols})"
