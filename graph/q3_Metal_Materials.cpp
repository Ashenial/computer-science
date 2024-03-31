///*The MIT License (MIT)
//
//Copyright (c) 2021-Present, Wencong Yang (yangwc3@mail2.sysu.edu.cn).
//
//Permission is hereby granted, free of charge, to any person obtaining a copy
//of this software and associated documentation files (the "Software"), to deal
//in the Software without restriction, including without limitation the rights
//to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
//copies of the Software, and to permit persons to whom the Software is
//furnished to do so, subject to the following conditions:
//
//The above copyright notice and this permission notice shall be included in
//all copies or substantial portions of the Software.
//
//THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
//IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
//AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
//THE SOFTWARE.*/
//
//#include <array>
//#include <vector>
//#include <thread>
//#include <iostream>
//
//#include "WindowsApp.h"
//#include "hittable_list.h"
//#include "material.h"
//#include "camera.h"
//static std::vector<std::vector<color>> gCanvas;		//Canvas
//
//// The width and height of the screen
//const auto aspect_ratio = 16.0 / 9.0;
//const int gWidth = 800;
//const int gHeight = static_cast<int>(gWidth / aspect_ratio);
//
//void rendering();
//color ray_color(const ray& r, const hittable& world, int depth);
//
//int main(int argc, char* args[])
//{
//	// Create window app handle
//	WindowsApp::ptr winApp = WindowsApp::getInstance(gWidth, gHeight, "CGAssignment4: Ray Tracing 21307359_LJH");
//	if (winApp == nullptr)
//	{
//		std::cerr << "Error: failed to create a window handler" << std::endl;
//		return -1;
//	}
//
//	// Memory allocation for canvas
//	gCanvas.resize(gHeight, std::vector<color>(gWidth));
//
//	// Launch the rendering thread
//	// Note: we run the rendering task in another thread to avoid GUI blocking
//	std::thread renderingThread(rendering);
//
//	// Window app loop
//	while (!winApp->shouldWindowClose())
//	{
//		// Process event
//		winApp->processEvent();
//
//		// Display to the screen
//		winApp->updateScreenSurface(gCanvas);
//
//	}
//
//	renderingThread.join();
//
//	return 0;
//}
//
//void write_color(int x, int y, color pixel_color, int samples_per_pixel)
//{
//	// Out-of-range detection
//	if (x < 0 || x >= gWidth)
//	{
//		std::cerr << "Warnning: try to write the pixel out of range: (x,y) -> (" << x << "," << y << ")" << std::endl;
//		return;
//	}
//
//	if (y < 0 || y >= gHeight)
//	{
//		std::cerr << "Warnning: try to write the pixel out of range: (x,y) -> (" << x << "," << y << ")" << std::endl;
//		return;
//	}
//
//	auto r = pixel_color.x();
//	auto g = pixel_color.y();
//	auto b = pixel_color.z();
//
//	// Divide the color by the number of samples and gamma-correct forgamma = 2.0.
//	auto scale = 1.0 / samples_per_pixel;
//	r = sqrt(scale * r);
//	g = sqrt(scale * g);
//	b = sqrt(scale * b);
//
//	r = clamp(r, 0.0, 0.999);
//	g = clamp(g, 0.0, 0.999);
//	b = clamp(b, 0.0, 0.999);
//
//	// Note: x -> the column number, y -> the row number
//	gCanvas[y][x] = vec3(r, g, b);
//
//}
//
//void rendering()
//{
//	double startFrame = clock();
//
//	printf("CGAssignment4 (built %s at %s) \n", __DATE__, __TIME__);
//	std::cout << "Ray-tracing based rendering launched..." << std::endl;
//
//	// Image
//
//	const int image_width = gWidth;
//	const int image_height = gHeight;
//	const int samples_per_pixel = 100;
//	const int max_depth = 50;
//
//	// World
//	hittable_list world;
//	auto material_ground = make_shared<lambertian>(color(0.8, 0.8, 0.0));
//	auto material_center = make_shared<lambertian>(color(0.7, 0.3, 0.3));
//	auto material_left = make_shared<metal>(color(0.8, 0.8, 0.8), 0.3);
//	auto material_right = make_shared<metal>(color(0.8, 0.6, 0.2), 1.0);
//	world.add(make_shared<sphere>(point3(0.0, -100.5, -1.0), 100.0,
//		material_ground));
//	world.add(make_shared<sphere>(point3(0.0, 0.0, -1.0), 0.5,
//		material_center));
//	world.add(make_shared<sphere>(point3(-1.0, 0.0, -1.0), 0.5,
//		material_left));
//	world.add(make_shared<sphere>(point3(1.0, 0.0, -1.0), 0.5,
//		material_right));
//
//	// Camera
//	camera cam;
//
//	// Render
//
//	for (int j = image_height - 1; j >= 0; --j) {
//		std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
//		for (int i = 0; i < image_width; ++i) {
//			color pixel_color(0, 0, 0);
//			for (int s = 0; s < samples_per_pixel; ++s) {
//				auto u = (i + random_double()) / (image_width - 1);
//				auto v = (j + random_double()) / (image_height - 1);
//				ray r = cam.get_ray(u, v);
//				pixel_color += ray_color(r, world, max_depth);
//			}
//			write_color(i, j, pixel_color, samples_per_pixel);
//		}
//	}
//
//	double endFrame = clock();
//	double timeConsuming = static_cast<double>(endFrame - startFrame) / CLOCKS_PER_SEC;
//	std::cout << "Ray-tracing based rendering over..." << std::endl;
//	std::cout << "The rendering task took " << timeConsuming << " seconds" << std::endl;
//}
//
//
//color ray_color(const ray& r, const hittable& world, int depth) {
//	hit_record rec;
//	// If we've exceeded the ray bounce limit, no more light is gathered.
//	if (depth <= 0)
//		return color(0, 0, 0);
//	if (world.hit(r, 0.001, infinity, rec)) {
//		ray scattered;
//		color attenuation;
//		if (rec.mat_ptr->scatter(r, rec, attenuation, scattered))
//			return attenuation * ray_color(scattered, world, depth - 1);
//		return color(0, 0, 0);
//	}
//	vec3 unit_direction = unit_vector(r.direction());
//	auto t = 0.5 * (unit_direction.y() + 1.0);
//	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
//}