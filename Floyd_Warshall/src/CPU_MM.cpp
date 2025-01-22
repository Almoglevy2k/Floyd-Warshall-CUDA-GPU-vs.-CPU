#include <iostream>
#include <stdexcept> // For std::invalid_argument
#include <iomanip> // For formatted output
#include <climits>  // For INT_MAX
/**
 * @brief Multiplies two matrices element-wise.
 *
 * This function performs CPU element-wise multiplication of two matrices `a` and `b`,
 * storing the result in the `result` matrix. All matrices are assumed to be square
 * matrices of size `n x n`.
 *
 * @param a Pointer to the first input matrix.
 * @param b Pointer to the second input matrix.
 * @param result Pointer to the output matrix where the result will be stored.
 * @param n The dimension of the matrices (number of rows/columns).
 */
void matrix_multiply(int *a, int *b, int *result, int n)
{
    // Null pointer checks
    if (a == nullptr || b == nullptr || result == nullptr)
    {
        throw std::invalid_argument("Null pointer passed to matrix_multiply");
    }

    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            result[i * n + j] = 0; // Initialize the result element
            for (int k = 0; k < n; k++)
            {
                result[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}
/**************************************************************************************************************************************** */

/**
 * @brief Prints a matrix to the console.
 *
 * @param matrix Pointer to the matrix.
 * @param n The dimension of the matrix (number of rows/columns).
 * @param name The name of the matrix to display.
 */
void print_matrix(const int *matrix, int n, const std::string &name)
{
    std::cout << "\n" << name << ":\n";
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            std::cout << std::setw(4) << matrix[i * n + j];
        }
        std::cout << std::endl;
    }
}

/**
 * @brief Compares two matrices and returns true if they are equal.
 *
 * @param result Pointer to the resulting matrix.
 * @param expected Pointer to the expected matrix.
 * @param n The dimension of the matrices (number of rows/columns).
 * @return true if the matrices are equal, false otherwise.
 */
bool validate_result(const int *result, const int *expected, int n)
{
    for (int i = 0; i < n * n; i++)
    {
        if (result[i] != expected[i])
        {
            return false;
        }
    }
    return true;
}

/*int main()
{
    const int n = 3; // Define the size of the matrices (3x3 in this case)

    // Initialize two 3x3 matrices
    int a[n][n] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}};
    int b[n][n] = {
        {1, 1, 1},
        {1, 1, 1},
        {1, 1, 1}};
    int result[n][n] = {0}; // Initialize result matrix with zeros

    // Expected result for validation
    int expected[n][n] = {
        {3, 3, 3},
        {3, 3, 3},
        {3, 3, 3}};

    // Perform matrix multiplication
    try
    {
        matrix_multiply(&a[0][0], &b[0][0], &result[0][0], n);
        std::cout << "Matrix multiplication was successful.\n";

        // Print the input matrices and the result
        print_matrix(&a[0][0], n, "Matrix A");
        print_matrix(&b[0][0], n, "Matrix B");
        print_matrix(&result[0][0], n, "Result of Matrix Multiplication");

        // Validate the result
        if (validate_result(&result[0][0], &expected[0][0], n))
        {
            std::cout << "\nThe result is correct!\n";
        }
        else
        {
            std::cout << "\nThe result is incorrect!\n";
            print_matrix(&expected[0][0], n, "Expected Result");
        }

        return 0; // Indicate success
    }
    catch (const std::exception &e)
    {
        std::cerr << "Matrix multiplication failed: " << e.what() << std::endl;
        return 1; // Indicate failure
    }
}*/
