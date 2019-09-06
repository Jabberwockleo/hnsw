/**
 * File              : main.cpp
 * Author            : Wan Li
 * Date              : 05.09.2019
 * Last Modified Date: 05.09.2019
 * Last Modified By  : Wan Li
 */

#include "../ann.h"
#include <iostream>

int main(int argc, char **argv) {
    ann::ANN ann("inner-product", 128);
    std::cout << "DONE" << std::endl;
    return 0;
}
