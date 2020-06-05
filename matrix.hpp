#pragma once

#include <assert.h>
#include <functional>
#include <iostream>
#include <math.h>
#include <stdlib.h>     /* srand, rand */
#include <time.h>       /* time */
#include <vector>

namespace matrix
{

    typedef std::vector<std::vector<float>> FloatMatrix;

    /*! return the number of rows in a matrix
     */
    int rows(const FloatMatrix& m)
    {
        return m.size();
    }

    /*! return the number of columns in a matrix
     */
    int cols(const FloatMatrix& m)
    {
        return m.size() == 0 ? 0 : m[0].size();
    }

    /*! return a matrix of specified dimensions, filled with zeroes
     */
    FloatMatrix zero(int rows, int cols)
    {
        assert(rows >= 0);
        assert(cols >= 0);
        FloatMatrix out;
        for(int i=0; i<rows; i++)
        {
            std::vector<float> row;
            for(int j=0; j<cols; j++)
            {
                row.push_back(0.0f);
            }
            out.push_back(row);
        }
        return out;
    }

    /*! return an identity matrix of specified dimensions
     */
    FloatMatrix eye(int rows, int cols)
    {
        assert(rows >= 0);
        assert(cols >= 0);
        auto out = zero(rows, cols);
        auto N = rows > cols ? rows : cols;
        for(int i=0; i<N; i++)
        {
            out[i][i] = 1.0f;
        }
        return out;
    }

    /*! return a matrix of specified dimensions, filled with elements from [0 .. 1]
     */
    FloatMatrix random(int rows, int cols)
    {
        assert(rows >= 0);
        assert(cols >= 0);
        srand(time(NULL));
        FloatMatrix out;
        for(int i=0; i<rows; i++)
        {
            std::vector<float> row;
            for(int j=0; j<cols; j++)
            {
                auto p = (rand() % 65536) / 65536.0f;
                row.push_back(p);
            }
            out.push_back(row);
        }
        return out;
    }

    /*! return the largest value from a given matrix
     */
    float max(const FloatMatrix& m)
    {
        assert(rows(m) > 0);
        assert(cols(m) > 0);
        auto out = m[0][0];
        for(int i=0; i<rows(m); i++)
        {
            for(int j=0; j<cols(m); j++)
            {
                out = m[i][j] > out ? m[i][j] : out;
            }
        }
        return out;
    }

    /*! return the smallest value from a given matrix
     */
    float min(const FloatMatrix& m)
    {
        assert(rows(m) > 0);
        assert(cols(m) > 0);
        auto out = m[0][0];
        for(int i=0; i<rows(m); i++)
        {
            for(int j=0; j<cols(m); j++)
            {
                out = m[i][j] < out ? m[i][j] : out;
            }
        }
        return out;
    }

    /*! add two matrices of the same dimensions
     */
    FloatMatrix add(const FloatMatrix& a, const FloatMatrix& b)
    {
        assert(rows(a) == rows(b));
        assert(cols(a) == cols(b));
        FloatMatrix out;
        for(int i=0; i<rows(a); i++)
        {
            std::vector<float> row;
            for(int j=0; j<cols(a); j++)
            {
                row.push_back(a[i][j] + b[i][j]);
            }
            out.push_back(row);
        }
        return out;
    }

    /* subtract two matrices of the same dimensions
     */
    FloatMatrix subtract(const FloatMatrix& a, const FloatMatrix& b)
    {
        assert(rows(a) == rows(b));
        assert(cols(a) == cols(b));
        FloatMatrix out;
        for(int i=0; i<rows(a); i++)
        {
            std::vector<float> row;
            for(int j=0; j<cols(a); j++)
            {
                row.push_back(a[i][j] - b[i][j]);
            }
            out.push_back(row);
        }
        return out;
    }

    /*! calculate the elementwise product of two matrices of the same dimensions
     */
    FloatMatrix dotproduct(const FloatMatrix& a, const FloatMatrix& b)
    {
        assert(rows(a) == rows(b));
        assert(cols(a) == cols(b));
        FloatMatrix out;
        for(int i=0; i<rows(a); i++)
        {
            std::vector<float> row;
            for(int j=0; j<cols(a); j++)
            {
                row.push_back(a[i][j] * b[i][j]);
            }
            out.push_back(row);
        }
        return out;
    }

    /*! multiply each element in a specified matrix with a specified scalar
     */
    FloatMatrix scalar(const FloatMatrix& a, float b)
    {
        assert(rows(a) > 0);
        assert(cols(a) > 0);
        FloatMatrix out;
        for(int i=0; i<rows(a); i++)
        {
            std::vector<float> row;
            for(int j=0; j<cols(a); j++)
            {
                row.push_back(a[i][j] * b);
            }
            out.push_back(row);
        }
        return out;
    }

    FloatMatrix transpose(const FloatMatrix& a)
    {
        assert(rows(a) > 0);
        assert(cols(a) > 0);
        FloatMatrix out;
        for(int i=0; i<cols(a); i++)
        {
            std::vector<float> row;
            for(int j=0; j<rows(a); j++)
            {
                row.push_back(a[j][i]);
            }
            out.push_back(row);
        }
        return out;
    }

    FloatMatrix mul(const FloatMatrix& a, const FloatMatrix& b)
    {
        assert(rows(a) > 0);
        assert(cols(a) > 0);
        assert(rows(b) > 0);
        assert(cols(b) > 0);
        assert(cols(a) == rows(b));
        auto out = zero(rows(a), cols(b));
        for(int i=0; i<rows(a); i++)
        {
            for(int j=0; j<cols(b); j++)
            {
                for(int k=0; k<cols(a); k++)
                {
                    out[i][j] += a[i][k] * b[k][j];
                }
            }
        }
        return out;
    }

    FloatMatrix apply_function(const FloatMatrix& a, const std::function<float(float)>& f)
    {
        auto out = FloatMatrix();
        for(int i=0; i<rows(a); i++)
        {
            std::vector<float> row;
            for(int j=0; j<cols(a); j++)
            {
                row.push_back(f(a[i][j]));
            }
            out.push_back(row);
        }
        return out;
    }

    void print_matrix(const FloatMatrix& m)
    {
        auto M = rows(m);
        auto N = cols(m);
        std::cout << "[";
        for(int i=0; i<M; i++)
        {
            std::cout << (i == 0 ? "" : " ") << "[";
            for(int j=0; j<N; j++)
            {
                std::cout << m[i][j] << (j == N - 1 ? "]" : " ");
            }
            std::cout << (i == M - 1 ? "]" : "") << std::endl;
        }
    }
}

