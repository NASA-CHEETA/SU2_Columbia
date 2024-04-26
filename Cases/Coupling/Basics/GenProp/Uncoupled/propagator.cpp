#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

int main() {
    const int n = 20;
    const double dn = 1.0 / n;
    const double dt = 0.01;
    double t = 0.0;

    // Generate mesh
    std::vector<double> x(n + 1);
    std::vector<double> y(n + 1);
    for (int i = 0; i <= n; ++i) {
        x[i] = i * dn;
        y[i] = i * dn;
    }

    // Initialize data on cell centers
    std::vector<std::vector<double>> u(n, std::vector<double>(n, 0));

    // Initial condition
    for (int j = 0; j < n; ++j) {
        u[j][n-1] = y[j];
    }

    // Prepare file for Tecplot
    std::ofstream tecplot_file("data.plt");
    tecplot_file << "TITLE = \"Visualization of Convection-Diffusion\"\n";
    tecplot_file << "VARIABLES = \"X\", \"Y\", \"U\"\n";
    tecplot_file << "ZONE T=\"Initial\", I=" << n << ", J=" << n << ", F=POINT\n";
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < n; ++i) {
            tecplot_file << x[i] << " " << y[j] << " " << u[j][i] << "\n";
        }
    }

    // Simulation loop
    while (true) {
        std::cout << "Propagating data..." << std::endl;

        // Convection-diffusion rule application
        // Top row
        for (int i = 0; i < n-1; ++i) {
            u[0][i] = 0.3 * u[0][i] + 0.6 * u[0][i+1] + 0.1 * u[1][i];
        }
        // Inner domain
        for (int j = 1; j < n-1; ++j) {
            for (int i = 0; i < n-1; ++i) {
                u[j][i] = 0.2 * u[j][i] + 0.6 * u[j][i+1] + 0.1 * u[j+1][i] + 0.1 * u[j-1][i];
            }
        }
        // Bottom row
        for (int i = 0; i < n-1; ++i) {
            u[n-1][i] = 0.3 * u[n-1][i] + 0.6 * u[n-1][i+1] + 0.1 * u[n-1][i];
        }

        // Output to Tecplot
        tecplot_file << "ZONE T=\"Time " << t << "\", I=" << n << ", J=" << n
                     << ", DATAPACKING=POINT, SOLUTIONTIME=" << t << std::endl;
        for (int j = 0; j < n; ++j) {
            for (int i = 0; i < n; ++i) {
                tecplot_file << x[i] << " " << y[j] << " " << u[j][i] << "\n";
            }
        }


        // Time advance
        t += dt;
        if (t > 0.2) {
            break;
        }
    }

    tecplot_file.close();
    std::cout << "Finished simulation and data output." << std::endl;

    return 0;
}
