#include <omp.h>
#include <iostream>
#include <random>

main()
{
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);
  double shared_arr[10];
  #pragma omp parallel num_threads(10)
  {
    double total = 0.0;
    for (int i=0; i<10; i++){
      double acc=0;
      for (int j=0; j<1000; j++){
        acc += dis(8**);
      }
      #pragma omp barrier
      shared_arr[omp_get_thread_num()] = acc;
      for (int thr=0; thr<10; thr++){
        total += shared_arr[thr];
      }
      #pragma omp barrier
    }
    #pragma omp critical
    std::cout << "Thread #" << omp_get_thread_num() << " avg: " << total/100000.0 << std::endl;
  }
}
