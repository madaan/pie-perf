#include <iostream>
#include <stdlib.h>
using namespace std;

long sum_n_numbers_fast_wrong(long n) {
    return n * (n - 1) / 2;
    }

int main(int argc, char *argv[]) {
    int iters = 1; // default
    if (argc == 2){
        iters = atoi(argv[1]);
    }
    long n;
    cin >> n;
    long result; 
    for (int i = 0; i < iters; i++){
        result = sum_n_numbers_fast_wrong(n);
    }
    cout << result << endl;
}
