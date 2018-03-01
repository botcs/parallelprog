#include <omp.h>
#include <iostream>

main()
{
  int x;
  x = 0;
  #pragma omp parallel shared(x)
  {
    #pragma omp critical
    x = x + 1;
  } /* end of parallel section */
  std::cout << x;
}
