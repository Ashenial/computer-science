﻿///*The MIT License (MIT)
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
//#include "vec3.h"
//#include "ray.h"
//
//static std::vector<std::vector<color>> gCanvas;		//Canvas
//
//// The width and height of the screen
//const auto aspect_ratio = 16.0 / 9.0;
//const int gWidth = 800;
//const int gHeight = static_cast<int>(gWidth / aspect_ratio);
//
//void rendering();
//color ray_color(const ray& r);
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
//void write_color(int x, int y, color pixel_color)
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
//	// Note: x -> the column number, y -> the row number
//	gCanvas[y][x] = pixel_color;
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
//
//	// Camera
//	auto viewport_height = 2.0;
//	auto viewport_width = aspect_ratio * viewport_height;
//	auto focal_length = 1.0;
//	auto origin = point3(0, 0, 0);
//	auto horizontal = vec3(viewport_width, 0, 0);
//	auto vertical = vec3(0, viewport_height, 0);
//	auto lower_left_corner = origin - horizontal / 2 - vertical / 2 - vec3(0, 0,
//		focal_length);
//
//	// Render
//
//	for (int j = image_height - 1; j >= 0; --j) {
//		std::cerr << "\rScanlines remaining: " << j << ' ' << std::flush;
//		for (int i = 0; i < image_width; ++i) {
//			auto u = double(i) / (image_width - 1);
//			auto v = double(j) / (image_height - 1);
//			ray r(origin, lower_left_corner + u * horizontal + v * vertical -
//				origin);
//			color pixel_color = ray_color(r);
//			write_color(i, j, pixel_color);
//		}
//	}
//
//	// The main ray-tracing based rendering loop, initial
//	// TODO: finish your own ray-tracing renderer according to the given tutorials
//	/*for (int j = image_height - 1; j >= 0; j--)
//	{
//		for (int i = 0; i < image_width; i++)
//		{
//			color pixel_color(double(i) / (image_width - 1),
//				double(j) / (image_height - 1), 0.25);
//			write_color(i, j, pixel_color);
//		}
//	}*/
//
//	double endFrame = clock();
//	double timeConsuming = static_cast<double>(endFrame - startFrame) / CLOCKS_PER_SEC;
//	std::cout << "Ray-tracing based rendering over..." << std::endl;
//	std::cout << "The rendering task took " << timeConsuming << " seconds" << std::endl;
//}
//
//color ray_color(const ray& r) {
//	vec3 unit_direction = unit_vector(r.direction());
//	auto t = 0.5 * (unit_direction.y() + 1.0);
//	return (1.0 - t) * color(1.0, 1.0, 1.0) + t * color(0.5, 0.7, 1.0);
//}