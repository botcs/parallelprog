#include <random>
#include <iostream>
#include <vector>
#include <chrono>
#include <algorithm>
#include <omp.h>
#include <iostream>

int main()
{
  //This is how we do timing with C++11
  std::chrono::time_point<std::chrono::system_clock> start, end;
  start = std::chrono::system_clock::now();
  int num_threads = 100;
  int total = 1000000;
  int inside = 0;
  #pragma omp parallel num_threads(num_threads)
  {

    //This is how we generate random numbers with C++11
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-0.5, -0.5);
    #pragma omp for reduction(+:inside)
    for (int i=0; i<total; i++){
      double x = dis(gen);
      double y = dis(gen);
      if ((x * x + y * y) <= 0.25){
        inside++;
      }
    }

  }
  double pi = 4.0 * (inside / double(total));
  //End timing
  end = std::chrono::system_clock::now();
  std::chrono::duration<double> elapsed_seconds = end-start;
  std::cout << "Pi approx: " << pi << std::endl;
  std::cout << "Runtime: " << elapsed_seconds.count() << "s\n";
}
