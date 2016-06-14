#include <iostream>

int main() {
	double r = 0.1, m = 1.0;
	int nx = 5, ny = 5, nz = 5;

	double mx = 2, my = 8, mz = 2;

	double x0 = (mx*2+1)*r;
	double y0 = (my*2+1)*r;
	double z0 = (mz*2+1)*r;

	//double xshift = r*0.05;
	//double zshift = r*0.1;
	double xshift = 0, zshift = 0;

	std::cout << nx*ny*nz << '\n';
	for (int ix = 0; ix < nx; ++ix)
		for (int iy = 0; iy < ny; ++iy)
			for (int iz = 0; iz < nz; ++iz) {
				std::cout << m << ' ' << r << ' ';
				std::cout << x0 + ix*2*r + xshift * iy << ' ' << y0 + iy*2*r << ' ' << z0 + iz*2*r + zshift*iy << ' ';
				std::cout << 0 << ' ' << 0 << ' ' << 0 << '\n';
			}

	std::cout << std::flush;

	double minX = 0, minY = 0, minZ = 0;
	double maxX = nx * 2 * r + 4*mx*r;
	double maxY = ny * 2 * r + 4*my*r;
	double maxZ = nz * 2 * r + 4*mz*r;
	std::cerr << "x_min " << minX << '\n';
	std::cerr << "y_min " << minY << '\n';
	std::cerr << "z_min " << minZ << '\n';
	std::cerr << "x_max " << maxX << '\n';
	std::cerr << "y_max " << maxY << '\n';
	std::cerr << "z_max " << maxZ << '\n';
	std::cerr << "x_reflect " << 1 << '\n';
	std::cerr << "y_reflect " << 1 << '\n';
	std::cerr << "z_reflect " << 1 << '\n';
	std::cerr << "x_g " << 0 << '\n';
	std::cerr << "y_g " << -1 << '\n';
	std::cerr << "z_g " << 0 << '\n';
	std::cerr << "x_n <= " << (maxX-minX)/(2*r) << '\n';
	std::cerr << "y_n <= " << (maxY-minY)/(2*r) << '\n';
	std::cerr << "z_n <= " << (maxZ-minZ)/(2*r) << '\n';
	std::cerr << std::flush;

	return 0;
}
