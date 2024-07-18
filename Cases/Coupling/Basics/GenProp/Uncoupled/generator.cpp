#include <iostream>
#include <Eigen/Dense>
#include <chrono>
#include <thread>
#include <random>

int main() {
    const int n = 20;
    const double dn = 1.0 / n;
    const double dt = 0.01;

    // Use Eigen library to create an evenly spaced array like np.linspace
    Eigen::VectorXd y = Eigen::VectorXd::LinSpaced(n+1, 0, 1);

    double t = 0.0;

    // Random number generation setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1, 1);

    while (true) {
        std::cout << "Generating data..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(200));

        // Generate random numbers like np.random.rand
        Eigen::VectorXd u(n);
        for (int i = 0; i < n; ++i) {
            u(i) = 1 - 2 * dis(gen);
        }

        // Advance time
        t += dt;

        if (t > 0.1) {
            std::cout << "Time window reached..Exiting!" << std::endl;
            break;
        }
    }

    return 0;
}