#include "../logistic_regression.hpp"

#include <iostream>
#include <vector>

void test_logistic_regression_001()
{


    /*
     * This dataset is originally from the National Institute of Diabetes and Digestive and Kidney Diseases.
     * The objective of the dataset is to diagnostically predict whether or not a patient has diabetes,
     * based on certain diagnostic measurements included in the dataset.
     * Several constraints were placed on the selection of these instances from a larger database.
     * In particular, all patients here are females at least 21 years old of Pima Indian heritage.
     */
    std::vector<std::vector<float>> xs =
    {
        {6, 148, 72, 35, 0, 33.6, 0.627, 50},
        {1, 85, 66, 29, 0, 26.6, 0.351, 31},
        {8, 183, 64, 0, 0, 23.3, 0.672, 32},
        {1, 89, 66, 23, 94, 28.1, 0.167, 21},
        {0, 137, 40, 35, 168, 43.1, 2.288, 33},
        {5, 116, 74, 0, 0, 25.6, 0.201, 30},
        {3, 78, 50, 32, 88, 31, 0.248, 26},
        {10, 115, 0, 0, 0, 35.3, 0.134, 29},
        {2, 197, 70, 45, 543, 30.5, 0.158, 53},
        {8, 125, 96, 0, 0, 0, 0.232, 54},
        {4, 110, 92, 0, 0, 37.6, 0.191, 30},
        {10, 168, 74, 0, 0, 38, 0.537, 34},
        {10, 139, 80, 0, 0, 27.1, 1.441, 57},
        {1, 189, 60, 23, 846, 30.1, 0.398, 59},
        {5, 166, 72, 19, 175, 25.8, 0.587, 51},
        {7, 100, 0, 0, 0, 30, 0.484, 32},
        {0, 118, 84, 47, 230, 45.8, 0.551, 31},
        {7, 107, 74, 0, 0, 29.6, 0.254, 31},
        {1, 103, 30, 38, 83, 43.3, 0.183, 33},
        {1, 115, 70, 30, 96, 34.6, 0.529, 32},
        {3, 126, 88, 41, 235, 39.3, 0.704, 27},
        {8, 99, 84, 0, 0, 35.4, 0.388, 50},
        {7, 196, 90, 0, 0, 39.8, 0.451, 41},
        {9, 119, 80, 35, 0, 29, 0.263, 29},
        {11, 143, 94, 33, 146, 36.6, 0.254, 51}
    };

    // scaling
    std::vector<float> scale;
    for(int i=0; i<xs[0].size(); i++)
    {
        scale.push_back(0.0f);
    }
    for(int i=0; i<xs.size(); i++)
    {
        for(int j=0; j<xs[i].size(); j++)
        {
            scale[j] = xs[i][j] > scale[j] ? xs[i][j] : scale[j];
        }
    }
    for(int i=0; i<xs.size(); i++)
    {
        for(int j=0; j<xs[i].size(); j++)
        {
            xs[i][j] /= scale[j];
        }
    }

    std::vector<float> ys =
    {
        1, 0, 1, 0, 1, 0, 1, 0, 1, 1,
        0, 1, 0, 1, 1, 1, 1, 1, 0, 1,
        0, 0, 1, 1, 1
    };

    auto coeffs = numeric::logistic_regression(xs, ys);
    std::cout << "Approx Coeffs. : {";
    for(int i=0; i<coeffs.size(); i++)
    {
        std::cout << coeffs[i] << ", ";
    }
    std::cout << "}" << std::endl;

    // test hypothesis
    auto tp = 0;
    auto tn = 0;
    auto fp = 0;
    auto fn = 0;
    std::cout << "Output hypothesis" << std::endl;
    for(int i=0; i<xs.size(); i++)
    {
        auto h = 0.0f;
        for(int j=0; j<xs[i].size(); j++)
        {
            h += xs[i][j] * coeffs[j];
        }
        h += coeffs[coeffs.size() -1];
        auto z = 1.0 / (1.0 + exp(-h));
        std::cout << h << "\t" << z << "\t" << ys[i] << std::endl;
        if(z >= 0.5)
        {
            if(ys[i] >= 0.5)
            {
                tp++;
            }
            else
            {
                fp++;
            }
        }
        else
        {
            if(ys[i] <= 0.5)
            {
                tn++;
            }
            else
            {
                fn++;
            }
        }
    }
    std::cout << "true positive : " << tp << ", false positive : " << fp << ", true negative : " << tn << ", false negative : " << fn << std::endl;

}

int main()
{
    test_logistic_regression_001();
}
