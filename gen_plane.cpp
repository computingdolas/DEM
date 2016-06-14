#include <iostream>
#include <limits>

struct vec3 {
	double v[3];

	vec3() {
		v[0] = v[1] = v[2] = 0;
	}

	vec3(double x, double y, double z) {
		v[0] = x;
		v[1] = y;
		v[2] = z;
	}

	double operator[](int i) const {
		return v[i];
	}

	const vec3 operator+(const vec3& x) const {
		return vec3(x[0] + v[0], x[1] + v[1], x[2] + v[2]);
	}

	const vec3 operator*(double a) const {
		return vec3(a * v[0], a * v[1], a * v[2]);
	}
};

int main() {
	double dy = 1.0;
	vec3 ll(1, 0.1+dy, 0.1), lu(1, 0.1+dy, 1.3), ul(0.1, 0.5+dy, 0.1), uu(0.1, 0.5+dy, 1.3);
	int numx = 32, numy = 32;
	double m = std::numeric_limits<double>::infinity();
	double r = 0.03;

	std::cout << numx * numy << '\n';
	for (int i = 0; i < numx; ++i) {
		for (int j = 0; j < numy; ++j) {
			double ax = (double)i / (numx - 1);
			double ay = (double)j / (numy - 1);
			
			vec3 x = (ul * ax + ll * (1-ax)) * ay + (uu * ax + lu * (1-ax)) * (1-ay);
			
			std::cout << m << ' ' << r << ' ' << x[0] << ' ' << x[1] << ' ' << x[2] << " 0 0 0\n";
		}
	}
	std::cout << std::flush;

	return 0;
}
