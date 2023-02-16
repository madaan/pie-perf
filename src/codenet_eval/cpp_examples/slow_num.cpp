#include <iostream>
#include <stdlib.h>

using namespace std;

long sum_n_numbers_slow(long n) {
    long sum = 0;
    for (int i = 0; i < n + 1; i++) {
        sum += i;
    }
    return sum;
    }


int main(int argc, char *argv[]) {
    int iters = 1; // default
    if (argc == 2){
        iters = atoi(argv[1]);
    }
    long n;
    cin >> n;
    // long result = sum_n_numbers_slow(n);
    long result;
    for (int i = 0; i < iters; i++){
        result = sum_n_numbers_slow(n);
    }
    cout << result << endl;
    // printf("%ld\n", result);
}