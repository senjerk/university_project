def transpose_matrix(matrix: list[list]) -> list[list]:
    transposed_matrix = [[0 for _ in range(len(matrix))] for _ in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            transposed_matrix[j][i] = matrix[i][j]
    return transposed_matrix


def multiplication_matrix(a_matrix: list[list], b_matrix: list[list]) -> list:
    m = len(a_matrix)
    n = len(b_matrix)
    k = len(b_matrix[0])

    result_matrix = [[None for __ in range(k)] for __ in range(m)]  # result_matrix: m × k

    for i in range(m):
        for j in range(k):
            result_matrix[i][j] = sum(a_matrix[i][kk] * b_matrix[kk][j] for kk in range(n))
    return result_matrix


def inverse_matrix(matrix: list[list]) -> list[list]:
    augmented_matrix = [
        [
            matrix[i][j] if j < len(matrix) else int(i == j - len(matrix))
            for j in range(2 * len(matrix))
        ]
        for i in range(len(matrix))
    ]
    for i in range(len(matrix)):
        pivot = augmented_matrix[i][i]
        if pivot == 0:
            raise ValueError("Matrix is not invertible")
        for j in range(2 * len(matrix)):
            augmented_matrix[i][j] /= pivot
        for j in range(len(matrix)):
            if i != j:
                scalar = augmented_matrix[j][i]
                for k in range(2 * len(matrix)):
                    augmented_matrix[j][k] -= scalar * augmented_matrix[i][k]
    inverse = [
        [augmented_matrix[i][j] for j in range(len(matrix), 2 * len(matrix))]
        for i in range(len(matrix))
    ]
    return inverse
