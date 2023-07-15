#include <omp.h>
#include <iostream>
#include <sstream>

const float R = 1.0;

float random(float, float);
void initialize(int, float, float, float*);
int monte_carlo(int, int, float*, float*);

int main(int argc, char* argv[]) {
    /* Precondition: argc == 2 */
    int n, t;
    std::istringstream iss_n(argv[1]);
    std::istringstream iss_t(argv[2]);
    iss_n >> n;
    iss_t >> t;

    /* Initialize the coordinate arrays */
    float* x = new float[n];
    float* y = new float[n];
    initialize(n, -R, R, x);
    initialize(n, -R, R, y);

    double start = omp_get_wtime();
    int in_circle = monte_carlo(n, t, x, y);
    double pi = 4. * in_circle / n;
    double end = omp_get_wtime();
    double duration = (end - start) * 1000.;
    std::cout << pi << "\n" << duration << std::endl;

    delete[] x;
    delete[] y;
    return 0;
}

/* Generate a random float between start and end, inclusive */
float random(float start, float end) {
    float delta = end - start;
    float r = (float) rand() / RAND_MAX * delta;
    return start + r;
}

/* Initialize the array arr */
void initialize(int n, float start, float end, float* arr) {
    int i;
    for (i = 0; i < n; i++) {
        arr[i] = random(start, end);
    }
}

/* Precondition: x and y have the same length */
int monte_carlo(int n, int t, float* x, float* y) {
    int i, count = 0;

    omp_set_num_threads(t);
    #pragma omp parallel for simd reduction(+:count)
    for (i = 0; i < n; i++) {
        int dummy = x[i] * x[i] + y[i] * y[i] <= R * R;
        count += dummy;
    }

    return count;
}