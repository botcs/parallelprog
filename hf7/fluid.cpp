#include <chrono>
#include <iostream>
#include <math.h>
#include <numeric>

// Global constants in the equations are
double rkold[3];
int nx1;
double rinv8;
double rinv9;
double Minf;
double rinv1;
double rinv4;
double rinv5;
double Pr;
double rinv12;
double rinv13;
double rinv10;
double rinv11;
double rknew[3];
double rc6;
double rc7;
double rc0;
double rc2;
double rc3;
int nx0;
double deltai1;
double deltai0;
double Re;
double deltat;
double gama;
int itercount;

// main program start
int main(int argc, char **argv) {

  // Initialising global constants
  nx0 = 257;
  if (argc > 1)
    nx0 = atoi(argv[1]);
  nx1 = 257;
  if (argc > 2)
    nx1 = atoi(argv[2]);
  gama = 1.40000000000000;
  Pr = 0.710000000000000;
  Re = 1600;
  deltat = 0.000846250000000000;
  Minf = 0.100000000000000;
  rc6 = 5.0 / 2.0;
  rc7 = 4.0 / 3.0;
  rc0 = 1.0 / 2.0;
  rc2 = 1.0 / 12.0;
  rc3 = 2.0 / 3.0;
  rkold[0] = 1.0 / 4.0;
  rkold[1] = 3.0 / 20.0;
  rkold[2] = 3.0 / 5.0;
  rknew[0] = 2.0 / 3.0;
  rknew[1] = 5.0 / 12.0;
  rknew[2] = 3.0 / 5.0;
  rinv12 = pow(Minf, -2);
  rinv13 = 1.0 / (gama * pow(Minf, 2));
  rinv10 = 1.0 / Pr;
  rinv11 = 1.0 / (gama - 1);
  rinv9 = 1.0 / Re;
  deltai1 = (1.0 / (nx1 - 1.0)) * M_PI;
  deltai0 = (1.0 / (nx0 - 1.0)) * M_PI;
  rinv8 = pow(deltai1, -2);
  rinv1 = 1.0 / deltai1;
  rinv4 = 1.0 / deltai0;
  rinv5 = pow(deltai0, -2);
  itercount = 20;

  std::cout << "Running on a " << nx0 + 4 << "x" << nx1 + 4 << " mesh for "
            << itercount << " iterations\n";

  // Allocating mesh
  double *restrict rho = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict rhou0 = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict rhou1 = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict rhoE = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict rho_old = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict rhou0_old = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict rhou1_old = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict rhoE_old = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict T = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict u0 = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict u1 = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict p = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict wk0 = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict wk1 = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict wk2 = new double[(nx0 + 4) * (nx1 + 4)];
  double *restrict wk3 = new double[(nx0 + 4) * (nx1 + 4)];

  // Initialisation
  //writing dataset rho with (i,j) access
  //writing dataset rhou0 with (i,j) access
  //writing dataset rhou1 with (i,j) access
  //writing dataset rhoE with (i,j) access
#pragma acc data copy(rho[0:nx0+4][0:nx1+4], rhou0[0:nx0+4][0:nx1+4], rhou1[0:nx0+4][0:nx1+4], rhoE[0:nx0+4][0:nx1+4])
{
  #pragma acc parallel loop collapse (2)
  for (int j = 0; j < nx1 + 4; j++) {
    for (int i = 0; i < nx0 + 4; i++) {
      double x = deltai0 * (i - 2);
      double y = deltai1 * (j - 2);
      double u = sin(x) * cos(y);
      double v = -cos(x) * sin(y);
      double p = 1.0 * rinv13 + 0.25 * (sin(2.0 * x) + sin(2.0 * y));
      double r = gama * pow(Minf, 2) * p;
      rho[(j + 0) * (nx0 + 4) + (i + 0)] = r;
      rhou0[(j + 0) * (nx0 + 4) + (i + 0)] = r * u;
      rhou1[(j + 0) * (nx0 + 4) + (i + 0)] = r * v;
      rhoE[(j + 0) * (nx0 + 4) + (i + 0)] =
          rinv11 * p + 0.5 * r * (pow(u, 2) + pow(v, 2));
    }
  }

  // Apply boundary conditions
  // Left
  //writing dataset rho with (i-1, j), (i-2, j) access
  //reading dataset rho with (i+1, j), (i+2, j) access
  //writing dataset rhou0 with (i-1, j), (i-2, j) access
  //reading dataset rhou0 with (i+1, j), (i+2, j) access
  //writing dataset rhou1 with (i-1, j), (i-2, j) access
  //reading dataset rhou1 with (i+1, j), (i+2, j) access
  //writing dataset rhoE with (i-1, j), (i-2, j) access
  //reading dataset rhoE with (i+1, j), (i+2, j) access
  #pragma acc parallel loop collapse (2)
  for (int j = 0; j < nx1 + 4; j++) {
    for (int i = 2; i < 3; i++) {
      rho[(j + 0) * (nx0 + 4) + (i - 1)] = rho[(j + 0) * (nx0 + 4) + (i + 1)];
      rho[(j + 0) * (nx0 + 4) + (i - 2)] = rho[(j + 0) * (nx0 + 4) + (i + 2)];
      rhou0[(j + 0) * (nx0 + 4) + (i - 1)] =
          rhou0[(j + 0) * (nx0 + 4) + (i + 1)];
      rhou0[(j + 0) * (nx0 + 4) + (i - 2)] =
          rhou0[(j + 0) * (nx0 + 4) + (i + 2)];
      rhou1[(j + 0) * (nx0 + 4) + (i - 1)] =
          rhou1[(j + 0) * (nx0 + 4) + (i + 1)];
      rhou1[(j + 0) * (nx0 + 4) + (i - 2)] =
          rhou1[(j + 0) * (nx0 + 4) + (i + 2)];
      rhoE[(j + 0) * (nx0 + 4) + (i - 1)] = rhoE[(j + 0) * (nx0 + 4) + (i + 1)];
      rhoE[(j + 0) * (nx0 + 4) + (i - 2)] = rhoE[(j + 0) * (nx0 + 4) + (i + 2)];
    }
  }

  // Right
  //writing dataset rho with (i+1, j), (i+2, j) access
  //reading dataset rho with (i-1, j), (i-2, j) access
  //writing dataset rhou0 with (i+1, j), (i+2, j) access
  //reading dataset rhou0 with (i-1, j), (i-2, j) access
  //writing dataset rhou1 with (i+1, j), (i+2, j) access
  //reading dataset rhou1 with (i-1, j), (i-2, j) access
  //writing dataset rhoE with (i+1, j), (i+2, j) access
  //reading dataset rhoE with (i-1, j), (i-2, j) access
  #pragma acc parallel loop collapse (2)
  for (int j = 0; j < nx1 + 4; j++) {
    for (int i = nx0 + 1; i < nx0 + 2; i++) {
      rho[(j + 0) * (nx0 + 4) + (i + 1)] = rho[(j + 0) * (nx0 + 4) + (i - 1)];
      rho[(j + 0) * (nx0 + 4) + (i + 2)] = rho[(j + 0) * (nx0 + 4) + (i - 2)];
      rhou0[(j + 0) * (nx0 + 4) + (i + 1)] =
          rhou0[(j + 0) * (nx0 + 4) + (i - 1)];
      rhou0[(j + 0) * (nx0 + 4) + (i + 2)] =
          rhou0[(j + 0) * (nx0 + 4) + (i - 2)];
      rhou1[(j + 0) * (nx0 + 4) + (i + 1)] =
          rhou1[(j + 0) * (nx0 + 4) + (i - 1)];
      rhou1[(j + 0) * (nx0 + 4) + (i + 2)] =
          rhou1[(j + 0) * (nx0 + 4) + (i - 2)];
      rhoE[(j + 0) * (nx0 + 4) + (i + 1)] = rhoE[(j + 0) * (nx0 + 4) + (i - 1)];
      rhoE[(j + 0) * (nx0 + 4) + (i + 2)] = rhoE[(j + 0) * (nx0 + 4) + (i - 2)];
    }
  }

  // Top
  //writing dataset rho with (i, j-1), (i, j-2) access
  //reading dataset rho with (i, j+1), (i, j+2) access
  //writing dataset rhou0 with (i, j-1), (i, j-2) access
  //reading dataset rhou0 with (i, j+1), (i, j+2) access
  //writing dataset rhou1 with (i, j-1), (i, j-2) access
  //reading dataset rhou1 with (i, j+1), (i, j+2) access
  //writing dataset rhoE with (i, j-1), (i, j-2) access
  //reading dataset rhoE with (i, j+1), (i, j+2) access
  #pragma acc parallel loop collapse (2)
  for (int j = 2; j < 3; j++) {
    for (int i = 0; i < nx0 + 4; i++) {
      rho[(j - 1) * (nx0 + 4) + (i + 0)] = rho[(j + 1) * (nx0 + 4) + (i + 0)];
      rho[(j - 2) * (nx0 + 4) + (i + 0)] = rho[(j + 2) * (nx0 + 4) + (i + 0)];
      rhou0[(j - 1) * (nx0 + 4) + (i + 0)] =
          rhou0[(j + 1) * (nx0 + 4) + (i + 0)];
      rhou0[(j - 2) * (nx0 + 4) + (i + 0)] =
          rhou0[(j + 2) * (nx0 + 4) + (i + 0)];
      rhou1[(j - 1) * (nx0 + 4) + (i + 0)] =
          rhou1[(j + 1) * (nx0 + 4) + (i + 0)];
      rhou1[(j - 2) * (nx0 + 4) + (i + 0)] =
          rhou1[(j + 2) * (nx0 + 4) + (i + 0)];
      rhoE[(j - 1) * (nx0 + 4) + (i + 0)] = rhoE[(j + 1) * (nx0 + 4) + (i + 0)];
      rhoE[(j - 2) * (nx0 + 4) + (i + 0)] = rhoE[(j + 2) * (nx0 + 4) + (i + 0)];
    }
  }

  // Bottom
  //writing dataset rho with (i, j+1), (i, j+2) access
  //reading dataset rho with (i, j-1), (i, j-2) access
  //writing dataset rhou0 with (i, j+1), (i, j+2) access
  //reading dataset rhou0 with (i, j-1), (i, j-2) access
  //writing dataset rhou1 with (i, j+1), (i, j+2) access
  //reading dataset rhou1 with (i, j-1), (i, j-2) access
  //writing dataset rhoE with (i, j+1), (i, j+2) access
  //reading dataset rhoE with (i, j-1), (i, j-2) access
  #pragma acc parallel loop collapse (2)
  for (int j = nx1 + 1; j < nx1 + 2; j++) {
    for (int i = 0; i < nx0 + 4; i++) {
      rho[(j + 1) * (nx0 + 4) + (i + 0)] = rho[(j - 1) * (nx0 + 4) + (i + 0)];
      rho[(j + 2) * (nx0 + 4) + (i + 0)] = rho[(j - 2) * (nx0 + 4) + (i + 0)];
      rhou0[(j + 1) * (nx0 + 4) + (i + 0)] =
          rhou0[(j - 1) * (nx0 + 4) + (i + 0)];
      rhou0[(j + 2) * (nx0 + 4) + (i + 0)] =
          rhou0[(j - 2) * (nx0 + 4) + (i + 0)];
      rhou1[(j + 1) * (nx0 + 4) + (i + 0)] =
          rhou1[(j - 1) * (nx0 + 4) + (i + 0)];
      rhou1[(j + 2) * (nx0 + 4) + (i + 0)] =
          rhou1[(j - 2) * (nx0 + 4) + (i + 0)];
      rhoE[(j + 1) * (nx0 + 4) + (i + 0)] = rhoE[(j - 1) * (nx0 + 4) + (i + 0)];
      rhoE[(j + 2) * (nx0 + 4) + (i + 0)] = rhoE[(j - 2) * (nx0 + 4) + (i + 0)];
    }
  }

} // pragma acc data close bracket

  // Record start time
  auto start = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> loop1 = std::chrono::high_resolution_clock::now() - start;
  std::chrono::duration<double> loop2 = std::chrono::high_resolution_clock::now() - start;
  std::chrono::duration<double> loop3 = std::chrono::high_resolution_clock::now() - start;
  std::chrono::duration<double> loop4 = std::chrono::high_resolution_clock::now() - start;
  std::chrono::duration<double> loop5 = std::chrono::high_resolution_clock::now() - start;
  std::chrono::duration<double> loop6 = std::chrono::high_resolution_clock::now() - start;
  std::chrono::duration<double> loop7 = std::chrono::high_resolution_clock::now() - start;
  std::chrono::duration<double> loop8 = std::chrono::high_resolution_clock::now() - start;
  std::chrono::duration<double> loop9 = std::chrono::high_resolution_clock::now() - start;
  std::chrono::duration<double> loop10 = std::chrono::high_resolution_clock::now() - start;

#pragma acc data copy(rho[0:nx0+4][0:nx1+4], rhou0[0:nx0+4][0:nx1+4], rhou1[0:nx0+4][0:nx1+4], rhoE[0:nx0+4][0:nx1+4]) 
#pragma acc data copy(rho_old[0:nx0+4][0:nx1+4], rhou0_old[0:nx0+4][0:nx1+4], rhou1_old[0:nx0+4][0:nx1+4], rhoE_old[0:nx0+4][0:nx1+4]) 
#pragma acc data copy(T[0:nx0+4][0:nx1+4], u0[0:nx0+4][0:nx1+4], u1[0:nx0+4][0:nx1+4], p[0:nx0+4][0:nx1+4]) 
#pragma acc data copy(wk0[0:nx0+4][0:nx1+4], wk1[0:nx0+4][0:nx1+4], wk2[0:nx0+4][0:nx1+4], wk3[0:nx0+4][0:nx1+4]) 
{
  // Main time iteration loop
  for (int iteration = 0; iteration < itercount; iteration++) {

    // Save equations
    //writing dataset rho_old with (i, j) access
    //reading dataset rho with (i, j) access
    //writing dataset rhou0_old with (i, j) access
    //reading dataset rhou0 with (i, j) access
    //writing dataset rhou1_old with (i, j) access
    //reading dataset rhou1 with (i, j) access
    //writing dataset rhoE_old with (i, j) access
    //reading dataset rhoE with (i, j) access
    auto start = std::chrono::high_resolution_clock::now();
    #pragma acc parallel loop collapse (2)
    for (int j = 0; j < nx1 + 4; j++) {
      for (int i = 0; i < nx0 + 4; i++) {
        rho_old[(j + 0) * (nx0 + 4) + (i + 0)] =
            rho[(j + 0) * (nx0 + 4) + (i + 0)];
        rhou0_old[(j + 0) * (nx0 + 4) + (i + 0)] =
            rhou0[(j + 0) * (nx0 + 4) + (i + 0)];
        rhou1_old[(j + 0) * (nx0 + 4) + (i + 0)] =
            rhou1[(j + 0) * (nx0 + 4) + (i + 0)];
        rhoE_old[(j + 0) * (nx0 + 4) + (i + 0)] =
            rhoE[(j + 0) * (nx0 + 4) + (i + 0)];
      }
    }
    loop1 += std::chrono::high_resolution_clock::now() - start;

    // Runge-Kutta time-stepper
    for (int stage = 0; stage < 3; stage++) {

      // Grouped Formula Evaluation
      //writing dataset T with (i, j) access
      //reading dataset rhou0 with (i, j) acces
      //reading dataset rhou1 with (i, j) acces
      //reading dataset rho with (i, j) acces
      //reading dataset rhoE with (i, j) acces
      //writing dataset p with (i, j) acces
      //writing dataset u1 with (i, j) acces
      //writing dataset u0 with (i, j) acces
      auto start = std::chrono::high_resolution_clock::now();
      #pragma acc parallel loop collapse (2)
      for (int j = 0; j < nx1 + 4; j++) {
        for (int i = 0; i < nx0 + 4; i++) {
          T[(j + 0) * (nx0 + 4) + (i + 0)] =
              gama * (gama - 1) *
              ((-rc0 * pow(rhou0[(j + 0) * (nx0 + 4) + (i + 0)], 2) -
                rc0 * pow(rhou1[(j + 0) * (nx0 + 4) + (i + 0)], 2)) /
                   rho[(j + 0) * (nx0 + 4) + (i + 0)] +
               rhoE[(j + 0) * (nx0 + 4) + (i + 0)]) *
              pow(Minf, 2) / rho[(j + 0) * (nx0 + 4) + (i + 0)];
          p[(j + 0) * (nx0 + 4) + (i + 0)] =
              (gama - 1) *
              ((-rc0 * pow(rhou0[(j + 0) * (nx0 + 4) + (i + 0)], 2) -
                rc0 * pow(rhou1[(j + 0) * (nx0 + 4) + (i + 0)], 2)) /
                   rho[(j + 0) * (nx0 + 4) + (i + 0)] +
               rhoE[(j + 0) * (nx0 + 4) + (i + 0)]);
          u1[(j + 0) * (nx0 + 4) + (i + 0)] =
              rhou1[(j + 0) * (nx0 + 4) + (i + 0)] /
              rho[(j + 0) * (nx0 + 4) + (i + 0)];
          u0[(j + 0) * (nx0 + 4) + (i + 0)] =
              rhou0[(j + 0) * (nx0 + 4) + (i + 0)] /
              rho[(j + 0) * (nx0 + 4) + (i + 0)];
        }
      }
      loop2 += std::chrono::high_resolution_clock::now() - start;
      // Residual of equation
      //writing dataset wk0 with (i,j) access
      //writing dataset wk1 with (i,j) access
      //writing dataset wk2 with (i,j) access
      //writing dataset wk3 with (i,j) access
      //reading dataset rhoE with (i,j), (i, j-2), (i,j-1) , (i,j+1), (i,j+2) access
      //reading dataset u1 with (i,j), (i, j-2), (i,j-1) , (i,j+1), (i,j+2) access
      //reading dataset u0 with (i,j), (i, j-2), (i,j-1) , (i,j+1), (i,j+2) access
      //reading dataset rhou1 with (i, j), (i-2, j), (i-1, j), (i+1, j), (i+2, j), (i, j-2), (i,j-1) , (i,j+1), (i,j+2) access
      //reading dataset rhou0 with (i, j), (i-2, j), (i-1, j), (i+1, j), (i+2, j), (i, j-2), (i,j-1) , (i,j+1), (i,j+2) access
      start = std::chrono::high_resolution_clock::now();
      #pragma acc parallel loop collapse (2)
      for (int j = 2; j < nx1 + 2; j++) {
        for (int i = 2; i < nx0 + 2; i++) {
          double temp_eval0 =
              rinv1 * ((rc2)*rhoE[(j - 2) * (nx0 + 4) + (i + 0)] *
                           u1[(j - 2) * (nx0 + 4) + (i + 0)] -
                       rc3 * rhoE[(j - 1) * (nx0 + 4) + (i + 0)] *
                           u1[(j - 1) * (nx0 + 4) + (i + 0)] +
                       (rc3)*rhoE[(j + 1) * (nx0 + 4) + (i + 0)] *
                           u1[(j + 1) * (nx0 + 4) + (i + 0)] -
                       rc2 * rhoE[(j + 2) * (nx0 + 4) + (i + 0)] *
                           u1[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval1 =
              rinv4 * ((rc2)*rhoE[(j + 0) * (nx0 + 4) + (i - 2)] -
                       rc3 * rhoE[(j + 0) * (nx0 + 4) + (i - 1)] +
                       (rc3)*rhoE[(j + 0) * (nx0 + 4) + (i + 1)] -
                       rc2 * rhoE[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval2 =
              rinv4 * ((rc2)*rhoE[(j + 0) * (nx0 + 4) + (i - 2)] *
                           u0[(j + 0) * (nx0 + 4) + (i - 2)] -
                       rc3 * rhoE[(j + 0) * (nx0 + 4) + (i - 1)] *
                           u0[(j + 0) * (nx0 + 4) + (i - 1)] +
                       (rc3)*rhoE[(j + 0) * (nx0 + 4) + (i + 1)] *
                           u0[(j + 0) * (nx0 + 4) + (i + 1)] -
                       rc2 * rhoE[(j + 0) * (nx0 + 4) + (i + 2)] *
                           u0[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval3 = rinv1 * ((rc2)*u1[(j - 2) * (nx0 + 4) + (i + 0)] -
                                       rc3 * u1[(j - 1) * (nx0 + 4) + (i + 0)] +
                                       (rc3)*u1[(j + 1) * (nx0 + 4) + (i + 0)] -
                                       rc2 * u1[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval4 = rinv5 * (-rc6 * T[(j + 0) * (nx0 + 4) + (i + 0)] -
                                       rc2 * T[(j + 0) * (nx0 + 4) + (i - 2)] +
                                       (rc7)*T[(j + 0) * (nx0 + 4) + (i - 1)] +
                                       (rc7)*T[(j + 0) * (nx0 + 4) + (i + 1)] -
                                       rc2 * T[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval5 =
              rinv1 * ((rc2)*rhou1[(j - 2) * (nx0 + 4) + (i + 0)] -
                       rc3 * rhou1[(j - 1) * (nx0 + 4) + (i + 0)] +
                       (rc3)*rhou1[(j + 1) * (nx0 + 4) + (i + 0)] -
                       rc2 * rhou1[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval6 =
              rinv5 * (-rc6 * u0[(j + 0) * (nx0 + 4) + (i + 0)] -
                       rc2 * u0[(j + 0) * (nx0 + 4) + (i - 2)] +
                       (rc7)*u0[(j + 0) * (nx0 + 4) + (i - 1)] +
                       (rc7)*u0[(j + 0) * (nx0 + 4) + (i + 1)] -
                       rc2 * u0[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval7 =
              rinv1 * ((rc2)*rhou1[(j - 2) * (nx0 + 4) + (i + 0)] *
                           u1[(j - 2) * (nx0 + 4) + (i + 0)] -
                       rc3 * rhou1[(j - 1) * (nx0 + 4) + (i + 0)] *
                           u1[(j - 1) * (nx0 + 4) + (i + 0)] +
                       (rc3)*rhou1[(j + 1) * (nx0 + 4) + (i + 0)] *
                           u1[(j + 1) * (nx0 + 4) + (i + 0)] -
                       rc2 * rhou1[(j + 2) * (nx0 + 4) + (i + 0)] *
                           u1[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval8 =
              rinv1 * ((rc2)*rho[(j - 2) * (nx0 + 4) + (i + 0)] -
                       rc3 * rho[(j - 1) * (nx0 + 4) + (i + 0)] +
                       (rc3)*rho[(j + 1) * (nx0 + 4) + (i + 0)] -
                       rc2 * rho[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval9 =
              rinv4 * ((rc2)*rhou0[(j + 0) * (nx0 + 4) + (i - 2)] -
                       rc3 * rhou0[(j + 0) * (nx0 + 4) + (i - 1)] +
                       (rc3)*rhou0[(j + 0) * (nx0 + 4) + (i + 1)] -
                       rc2 * rhou0[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval10 =
              rinv1 * ((rc2)*u0[(j - 2) * (nx0 + 4) + (i + 0)] -
                       rc3 * u0[(j - 1) * (nx0 + 4) + (i + 0)] +
                       (rc3)*u0[(j + 1) * (nx0 + 4) + (i + 0)] -
                       rc2 * u0[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval11 = rinv1 * ((rc2)*p[(j - 2) * (nx0 + 4) + (i + 0)] -
                                        rc3 * p[(j - 1) * (nx0 + 4) + (i + 0)] +
                                        (rc3)*p[(j + 1) * (nx0 + 4) + (i + 0)] -
                                        rc2 * p[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval12 =
              rinv4 * ((rc2)*u1[(j + 0) * (nx0 + 4) + (i - 2)] -
                       rc3 * u1[(j + 0) * (nx0 + 4) + (i - 1)] +
                       (rc3)*u1[(j + 0) * (nx0 + 4) + (i + 1)] -
                       rc2 * u1[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval13 =
              rinv4 * ((rc2)*rhou1[(j + 0) * (nx0 + 4) + (i - 2)] *
                           u0[(j + 0) * (nx0 + 4) + (i - 2)] -
                       rc3 * rhou1[(j + 0) * (nx0 + 4) + (i - 1)] *
                           u0[(j + 0) * (nx0 + 4) + (i - 1)] +
                       (rc3)*rhou1[(j + 0) * (nx0 + 4) + (i + 1)] *
                           u0[(j + 0) * (nx0 + 4) + (i + 1)] -
                       rc2 * rhou1[(j + 0) * (nx0 + 4) + (i + 2)] *
                           u0[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval14 =
              rinv4 * ((rc2)*rho[(j + 0) * (nx0 + 4) + (i - 2)] *
                           u0[(j + 0) * (nx0 + 4) + (i - 2)] -
                       rc3 * rho[(j + 0) * (nx0 + 4) + (i - 1)] *
                           u0[(j + 0) * (nx0 + 4) + (i - 1)] +
                       (rc3)*rho[(j + 0) * (nx0 + 4) + (i + 1)] *
                           u0[(j + 0) * (nx0 + 4) + (i + 1)] -
                       rc2 * rho[(j + 0) * (nx0 + 4) + (i + 2)] *
                           u0[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval15 =
              rinv4 * ((rc2)*rho[(j + 0) * (nx0 + 4) + (i - 2)] -
                       rc3 * rho[(j + 0) * (nx0 + 4) + (i - 1)] +
                       (rc3)*rho[(j + 0) * (nx0 + 4) + (i + 1)] -
                       rc2 * rho[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval16 =
              rinv1 * ((rc2)*rhou0[(j - 2) * (nx0 + 4) + (i + 0)] -
                       rc3 * rhou0[(j - 1) * (nx0 + 4) + (i + 0)] +
                       (rc3)*rhou0[(j + 1) * (nx0 + 4) + (i + 0)] -
                       rc2 * rhou0[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval17 =
              rinv5 * (-rc6 * u1[(j + 0) * (nx0 + 4) + (i + 0)] -
                       rc2 * u1[(j + 0) * (nx0 + 4) + (i - 2)] +
                       (rc7)*u1[(j + 0) * (nx0 + 4) + (i - 1)] +
                       (rc7)*u1[(j + 0) * (nx0 + 4) + (i + 1)] -
                       rc2 * u1[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval18 =
              rinv4 * ((rc2)*rhou0[(j + 0) * (nx0 + 4) + (i - 2)] *
                           u0[(j + 0) * (nx0 + 4) + (i - 2)] -
                       rc3 * rhou0[(j + 0) * (nx0 + 4) + (i - 1)] *
                           u0[(j + 0) * (nx0 + 4) + (i - 1)] +
                       (rc3)*rhou0[(j + 0) * (nx0 + 4) + (i + 1)] *
                           u0[(j + 0) * (nx0 + 4) + (i + 1)] -
                       rc2 * rhou0[(j + 0) * (nx0 + 4) + (i + 2)] *
                           u0[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval19 =
              rinv1 * ((rc2)*rhou0[(j - 2) * (nx0 + 4) + (i + 0)] *
                           u1[(j - 2) * (nx0 + 4) + (i + 0)] -
                       rc3 * rhou0[(j - 1) * (nx0 + 4) + (i + 0)] *
                           u1[(j - 1) * (nx0 + 4) + (i + 0)] +
                       (rc3)*rhou0[(j + 1) * (nx0 + 4) + (i + 0)] *
                           u1[(j + 1) * (nx0 + 4) + (i + 0)] -
                       rc2 * rhou0[(j + 2) * (nx0 + 4) + (i + 0)] *
                           u1[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval20 =
              rinv8 * (-rc6 * u1[(j + 0) * (nx0 + 4) + (i + 0)] -
                       rc2 * u1[(j - 2) * (nx0 + 4) + (i + 0)] +
                       (rc7)*u1[(j - 1) * (nx0 + 4) + (i + 0)] +
                       (rc7)*u1[(j + 1) * (nx0 + 4) + (i + 0)] -
                       rc2 * u1[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval21 = rinv4 * ((rc2)*p[(j + 0) * (nx0 + 4) + (i - 2)] -
                                        rc3 * p[(j + 0) * (nx0 + 4) + (i - 1)] +
                                        (rc3)*p[(j + 0) * (nx0 + 4) + (i + 1)] -
                                        rc2 * p[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval22 =
              rinv8 * (-rc6 * u0[(j + 0) * (nx0 + 4) + (i + 0)] -
                       rc2 * u0[(j - 2) * (nx0 + 4) + (i + 0)] +
                       (rc7)*u0[(j - 1) * (nx0 + 4) + (i + 0)] +
                       (rc7)*u0[(j + 1) * (nx0 + 4) + (i + 0)] -
                       rc2 * u0[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval23 =
              rinv1 * ((rc2)*rhoE[(j - 2) * (nx0 + 4) + (i + 0)] -
                       rc3 * rhoE[(j - 1) * (nx0 + 4) + (i + 0)] +
                       (rc3)*rhoE[(j + 1) * (nx0 + 4) + (i + 0)] -
                       rc2 * rhoE[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval24 =
              rinv8 * (-rc6 * T[(j + 0) * (nx0 + 4) + (i + 0)] -
                       rc2 * T[(j - 2) * (nx0 + 4) + (i + 0)] +
                       (rc7)*T[(j - 1) * (nx0 + 4) + (i + 0)] +
                       (rc7)*T[(j + 1) * (nx0 + 4) + (i + 0)] -
                       rc2 * T[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval25 =
              rinv4 * ((rc2)*rhou1[(j + 0) * (nx0 + 4) + (i - 2)] -
                       rc3 * rhou1[(j + 0) * (nx0 + 4) + (i - 1)] +
                       (rc3)*rhou1[(j + 0) * (nx0 + 4) + (i + 1)] -
                       rc2 * rhou1[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval26 = rinv4 * ((rc2)*p[(j + 0) * (nx0 + 4) + (i - 2)] *
                                            u0[(j + 0) * (nx0 + 4) + (i - 2)] -
                                        rc3 * p[(j + 0) * (nx0 + 4) + (i - 1)] *
                                            u0[(j + 0) * (nx0 + 4) + (i - 1)] +
                                        (rc3)*p[(j + 0) * (nx0 + 4) + (i + 1)] *
                                            u0[(j + 0) * (nx0 + 4) + (i + 1)] -
                                        rc2 * p[(j + 0) * (nx0 + 4) + (i + 2)] *
                                            u0[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval27 =
              rinv4 * ((rc2)*u0[(j + 0) * (nx0 + 4) + (i - 2)] -
                       rc3 * u0[(j + 0) * (nx0 + 4) + (i - 1)] +
                       (rc3)*u0[(j + 0) * (nx0 + 4) + (i + 1)] -
                       rc2 * u0[(j + 0) * (nx0 + 4) + (i + 2)]);
          double temp_eval28 = rinv1 * ((rc2)*p[(j - 2) * (nx0 + 4) + (i + 0)] *
                                            u1[(j - 2) * (nx0 + 4) + (i + 0)] -
                                        rc3 * p[(j - 1) * (nx0 + 4) + (i + 0)] *
                                            u1[(j - 1) * (nx0 + 4) + (i + 0)] +
                                        (rc3)*p[(j + 1) * (nx0 + 4) + (i + 0)] *
                                            u1[(j + 1) * (nx0 + 4) + (i + 0)] -
                                        rc2 * p[(j + 2) * (nx0 + 4) + (i + 0)] *
                                            u1[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval29 =
              rinv1 * ((rc2)*rho[(j - 2) * (nx0 + 4) + (i + 0)] *
                           u1[(j - 2) * (nx0 + 4) + (i + 0)] -
                       rc3 * rho[(j - 1) * (nx0 + 4) + (i + 0)] *
                           u1[(j - 1) * (nx0 + 4) + (i + 0)] +
                       (rc3)*rho[(j + 1) * (nx0 + 4) + (i + 0)] *
                           u1[(j + 1) * (nx0 + 4) + (i + 0)] -
                       rc2 * rho[(j + 2) * (nx0 + 4) + (i + 0)] *
                           u1[(j + 2) * (nx0 + 4) + (i + 0)]);
          double temp_eval30 =
              rinv1 * ((rc2)*rinv4 * ((rc2)*u0[(j - 2) * (nx0 + 4) + (i - 2)] -
                                      rc3 * u0[(j - 2) * (nx0 + 4) + (i - 1)] +
                                      (rc3)*u0[(j - 2) * (nx0 + 4) + (i + 1)] -
                                      rc2 * u0[(j - 2) * (nx0 + 4) + (i + 2)]) -
                       rc3 * rinv4 * ((rc2)*u0[(j - 1) * (nx0 + 4) + (i - 2)] -
                                      rc3 * u0[(j - 1) * (nx0 + 4) + (i - 1)] +
                                      (rc3)*u0[(j - 1) * (nx0 + 4) + (i + 1)] -
                                      rc2 * u0[(j - 1) * (nx0 + 4) + (i + 2)]) +
                       (rc3)*rinv4 * ((rc2)*u0[(j + 1) * (nx0 + 4) + (i - 2)] -
                                      rc3 * u0[(j + 1) * (nx0 + 4) + (i - 1)] +
                                      (rc3)*u0[(j + 1) * (nx0 + 4) + (i + 1)] -
                                      rc2 * u0[(j + 1) * (nx0 + 4) + (i + 2)]) -
                       rc2 * rinv4 * ((rc2)*u0[(j + 2) * (nx0 + 4) + (i - 2)] -
                                      rc3 * u0[(j + 2) * (nx0 + 4) + (i - 1)] +
                                      (rc3)*u0[(j + 2) * (nx0 + 4) + (i + 1)] -
                                      rc2 * u0[(j + 2) * (nx0 + 4) + (i + 2)]));
          double temp_eval31 =
              rinv1 * ((rc2)*rinv4 * ((rc2)*u1[(j - 2) * (nx0 + 4) + (i - 2)] -
                                      rc3 * u1[(j - 2) * (nx0 + 4) + (i - 1)] +
                                      (rc3)*u1[(j - 2) * (nx0 + 4) + (i + 1)] -
                                      rc2 * u1[(j - 2) * (nx0 + 4) + (i + 2)]) -
                       rc3 * rinv4 * ((rc2)*u1[(j - 1) * (nx0 + 4) + (i - 2)] -
                                      rc3 * u1[(j - 1) * (nx0 + 4) + (i - 1)] +
                                      (rc3)*u1[(j - 1) * (nx0 + 4) + (i + 1)] -
                                      rc2 * u1[(j - 1) * (nx0 + 4) + (i + 2)]) +
                       (rc3)*rinv4 * ((rc2)*u1[(j + 1) * (nx0 + 4) + (i - 2)] -
                                      rc3 * u1[(j + 1) * (nx0 + 4) + (i - 1)] +
                                      (rc3)*u1[(j + 1) * (nx0 + 4) + (i + 1)] -
                                      rc2 * u1[(j + 1) * (nx0 + 4) + (i + 2)]) -
                       rc2 * rinv4 * ((rc2)*u1[(j + 2) * (nx0 + 4) + (i - 2)] -
                                      rc3 * u1[(j + 2) * (nx0 + 4) + (i - 1)] +
                                      (rc3)*u1[(j + 2) * (nx0 + 4) + (i + 1)] -
                                      rc2 * u1[(j + 2) * (nx0 + 4) + (i + 2)]));
          wk0[(j + 0) * (nx0 + 4) + (i + 0)] =
              -0.5 * temp_eval14 -
              0.5 * temp_eval15 * u0[(j + 0) * (nx0 + 4) + (i + 0)] -
              0.5 * temp_eval29 -
              0.5 * temp_eval8 * u1[(j + 0) * (nx0 + 4) + (i + 0)] -
              0.5 * (temp_eval27 + temp_eval3) *
                  rho[(j + 0) * (nx0 + 4) + (i + 0)];
          wk1[(j + 0) * (nx0 + 4) + (i + 0)] =
              -0.5 * temp_eval16 * u1[(j + 0) * (nx0 + 4) + (i + 0)] -
              0.5 * temp_eval18 - 0.5 * temp_eval19 - temp_eval21 -
              0.5 * temp_eval9 * u0[(j + 0) * (nx0 + 4) + (i + 0)] +
              rinv9 * (temp_eval22 + temp_eval31) +
              rinv9 * (-rc3 * temp_eval31 + (rc7)*temp_eval6) -
              0.5 * (temp_eval27 + temp_eval3) *
                  rhou0[(j + 0) * (nx0 + 4) + (i + 0)];
          wk2[(j + 0) * (nx0 + 4) + (i + 0)] =
              -temp_eval11 - 0.5 * temp_eval13 -
              0.5 * temp_eval25 * u0[(j + 0) * (nx0 + 4) + (i + 0)] -
              0.5 * temp_eval5 * u1[(j + 0) * (nx0 + 4) + (i + 0)] -
              0.5 * temp_eval7 + rinv9 * (temp_eval17 + temp_eval30) +
              rinv9 * ((rc7)*temp_eval20 - rc3 * temp_eval30) -
              0.5 * (temp_eval27 + temp_eval3) *
                  rhou1[(j + 0) * (nx0 + 4) + (i + 0)];
          wk3[(j + 0) * (nx0 + 4) + (i + 0)] =
              -0.5 * temp_eval0 -
              0.5 * temp_eval1 * u0[(j + 0) * (nx0 + 4) + (i + 0)] +
              temp_eval10 * rinv9 * (temp_eval10 + temp_eval12) +
              temp_eval12 * rinv9 * (temp_eval10 + temp_eval12) -
              0.5 * temp_eval2 -
              0.5 * temp_eval23 * u1[(j + 0) * (nx0 + 4) + (i + 0)] +
              temp_eval24 * rinv10 * rinv11 * rinv12 * rinv9 - temp_eval26 +
              temp_eval27 * rinv9 * ((rc7)*temp_eval27 - rc3 * temp_eval3) -
              temp_eval28 +
              temp_eval3 * rinv9 * (-rc3 * temp_eval27 + (rc7)*temp_eval3) +
              temp_eval4 * rinv10 * rinv11 * rinv12 * rinv9 +
              rinv9 * (temp_eval17 + temp_eval30) *
                  u1[(j + 0) * (nx0 + 4) + (i + 0)] +
              rinv9 * ((rc7)*temp_eval20 - rc3 * temp_eval30) *
                  u1[(j + 0) * (nx0 + 4) + (i + 0)] +
              rinv9 * (temp_eval22 + temp_eval31) *
                  u0[(j + 0) * (nx0 + 4) + (i + 0)] +
              rinv9 * (-rc3 * temp_eval31 + (rc7)*temp_eval6) *
                  u0[(j + 0) * (nx0 + 4) + (i + 0)] -
              0.5 * (temp_eval27 + temp_eval3) *
                  rhoE[(j + 0) * (nx0 + 4) + (i + 0)];
        }
      }
      loop3 += std::chrono::high_resolution_clock::now() - start;
      // RK new (subloop) update
      //writing dataset rho with (i, j) access
      //reading dataset wk0 with (i, j) access
      //reading dataset rho_old with (i, j) access
      //writing dataset rhou0 with (i, j) access
      //reading dataset wk1 with (i, j) access
      //reading dataset rhou0_old with (i, j) access
      //writing dataset rhou1 with (i, j) access
      //reading dataset wk2 with (i, j) access
      //reading dataset rhou1_old with (i, j) access
      //writing dataset rhoE with (i, j) access
      //reading dataset wk3 with (i, j) access
      //reading dataset rhoE_old with (i, j) access
      start = std::chrono::high_resolution_clock::now();
      #pragma acc parallel loop collapse (2)
      for (int j = 0; j < nx1 + 4; j++) {
        for (int i = 0; i < nx0 + 4; i++) {
          rho[(j + 0) * (nx0 + 4) + (i + 0)] =
              deltat * rknew[0] * wk0[(j + 0) * (nx0 + 4) + (i + 0)] +
              rho_old[(j + 0) * (nx0 + 4) + (i + 0)];
          rhou0[(j + 0) * (nx0 + 4) + (i + 0)] =
              deltat * rknew[0] * wk1[(j + 0) * (nx0 + 4) + (i + 0)] +
              rhou0_old[(j + 0) * (nx0 + 4) + (i + 0)];
          rhou1[(j + 0) * (nx0 + 4) + (i + 0)] =
              deltat * rknew[0] * wk2[(j + 0) * (nx0 + 4) + (i + 0)] +
              rhou1_old[(j + 0) * (nx0 + 4) + (i + 0)];
          rhoE[(j + 0) * (nx0 + 4) + (i + 0)] =
              deltat * rknew[0] * wk3[(j + 0) * (nx0 + 4) + (i + 0)] +
              rhoE_old[(j + 0) * (nx0 + 4) + (i + 0)];
        }
      }
      loop4 += std::chrono::high_resolution_clock::now() - start;
      // RK old update
      //writing rho_old with (i, j) access
      //reading wk0 with (i, j) access
      //reading rho_old with (i, j) access
      //writing rhou0_old with (i, j) access
      //reading wk1 with (i, j) access
      //reading rho_old with (i, j) access
      //writing rhou1_old with (i, j) access
      //reading wk2 with (i, j) access
      //reading rho_old with (i, j) access
      //writing rhoE_old with (i, j) access
      //reading wk3 with (i, j) access
      //reading rhoE_old with (i, j) access
      start = std::chrono::high_resolution_clock::now();
      #pragma acc parallel loop collapse (2)
      for (int j = 0; j < nx1 + 4; j++) {
        for (int i = 0; i < nx0 + 4; i++) {
          rho_old[(j + 0) * (nx0 + 4) + (i + 0)] =
              deltat * rkold[0] * wk0[(j + 0) * (nx0 + 4) + (i + 0)] +
              rho_old[(j + 0) * (nx0 + 4) + (i + 0)];
          rhou0_old[(j + 0) * (nx0 + 4) + (i + 0)] =
              deltat * rkold[0] * wk1[(j + 0) * (nx0 + 4) + (i + 0)] +
              rhou0_old[(j + 0) * (nx0 + 4) + (i + 0)];
          rhou1_old[(j + 0) * (nx0 + 4) + (i + 0)] =
              deltat * rkold[0] * wk2[(j + 0) * (nx0 + 4) + (i + 0)] +
              rhou1_old[(j + 0) * (nx0 + 4) + (i + 0)];
          rhoE_old[(j + 0) * (nx0 + 4) + (i + 0)] =
              deltat * rkold[0] * wk3[(j + 0) * (nx0 + 4) + (i + 0)] +
              rhoE_old[(j + 0) * (nx0 + 4) + (i + 0)];
        }
      }
      loop5 += std::chrono::high_resolution_clock::now() - start;
      // Apply boundary conditions

      // Left
      //writing dataset rho with (i-1, j), (i-2, j) access
      //reading dataset rho with (i+1, j), (i+2, j) access
      //writing dataset rhou0 with (i-1, j), (i-2, j) access
      //reading dataset rhou0 with (i+1, j), (i+2, j) access
      //writing dataset rhou1 with (i-1, j), (i-2, j) access
      //reading dataset rhou1 with (i+1, j), (i+2, j) access
      //writing dataset rhoE with (i-1, j), (i-2, j) access
      //reading dataset rhoE with (i+1, j), (i+2, j) access
      start = std::chrono::high_resolution_clock::now();
      #pragma acc parallel loop collapse (2)
      for (int j = 0; j < nx1 + 4; j++) {
        for (int i = 2; i < 3; i++) {
          rho[(j + 0) * (nx0 + 4) + (i - 1)] =
              rho[(j + 0) * (nx0 + 4) + (i + 1)];
          rho[(j + 0) * (nx0 + 4) + (i - 2)] =
              rho[(j + 0) * (nx0 + 4) + (i + 2)];
          rhou0[(j + 0) * (nx0 + 4) + (i - 1)] =
              rhou0[(j + 0) * (nx0 + 4) + (i + 1)];
          rhou0[(j + 0) * (nx0 + 4) + (i - 2)] =
              rhou0[(j + 0) * (nx0 + 4) + (i + 2)];
          rhou1[(j + 0) * (nx0 + 4) + (i - 1)] =
              rhou1[(j + 0) * (nx0 + 4) + (i + 1)];
          rhou1[(j + 0) * (nx0 + 4) + (i - 2)] =
              rhou1[(j + 0) * (nx0 + 4) + (i + 2)];
          rhoE[(j + 0) * (nx0 + 4) + (i - 1)] =
              rhoE[(j + 0) * (nx0 + 4) + (i + 1)];
          rhoE[(j + 0) * (nx0 + 4) + (i - 2)] =
              rhoE[(j + 0) * (nx0 + 4) + (i + 2)];
        }
      }
      loop6 += std::chrono::high_resolution_clock::now() - start;

      // Right
      //writing dataset rho with (i+1, j), (i+2, j) access
      //reading dataset rho with (i-1, j), (i-2, j) access
      //writing dataset rhou0 with (i+1, j), (i+2, j) access
      //reading dataset rhou0 with (i-1, j), (i-2, j) access
      //writing dataset rhou1 with (i+1, j), (i+2, j) access
      //reading dataset rhou1 with (i-1, j), (i-2, j) access
      //writing dataset rhoE with (i+1, j), (i+2, j) access
      //reading dataset rhoE with (i-1, j), (i-2, j) access
      start = std::chrono::high_resolution_clock::now();
      #pragma acc parallel loop collapse (2)
      for (int j = 0; j < nx1 + 4; j++) {
        for (int i = nx0 + 1; i < nx0 + 2; i++) {
          rho[(j + 0) * (nx0 + 4) + (i + 1)] =
              rho[(j + 0) * (nx0 + 4) + (i - 1)];
          rho[(j + 0) * (nx0 + 4) + (i + 2)] =
              rho[(j + 0) * (nx0 + 4) + (i - 2)];
          rhou0[(j + 0) * (nx0 + 4) + (i + 1)] =
              rhou0[(j + 0) * (nx0 + 4) + (i - 1)];
          rhou0[(j + 0) * (nx0 + 4) + (i + 2)] =
              rhou0[(j + 0) * (nx0 + 4) + (i - 2)];
          rhou1[(j + 0) * (nx0 + 4) + (i + 1)] =
              rhou1[(j + 0) * (nx0 + 4) + (i - 1)];
          rhou1[(j + 0) * (nx0 + 4) + (i + 2)] =
              rhou1[(j + 0) * (nx0 + 4) + (i - 2)];
          rhoE[(j + 0) * (nx0 + 4) + (i + 1)] =
              rhoE[(j + 0) * (nx0 + 4) + (i - 1)];
          rhoE[(j + 0) * (nx0 + 4) + (i + 2)] =
              rhoE[(j + 0) * (nx0 + 4) + (i - 2)];
        }
      }
      loop7 += std::chrono::high_resolution_clock::now() - start;
      // Top
      //writing dataset rho with (i, j-1), (i, j-2) access
      //reading dataset rho with (i, j+1), (i, j+2) access
      //writing dataset rhou0 with (i, j-1), (i, j-2) access
      //reading dataset rhou0 with (i, j+1), (i, j+2) access
      //writing dataset rhou1 with (i, j-1), (i, j-2) access
      //reading dataset rhou1 with (i, j+1), (i, j+2) access
      //writing dataset rhoE with (i, j-1), (i, j-2) access
      //reading dataset rhoE with (i, j+1), (i, j+2) access
      start = std::chrono::high_resolution_clock::now();
      #pragma acc parallel loop collapse (2)
      for (int j = 2; j < 3; j++) {
        for (int i = 0; i < nx0 + 4; i++) {
          rho[(j - 1) * (nx0 + 4) + (i + 0)] =
              rho[(j + 1) * (nx0 + 4) + (i + 0)];
          rho[(j - 2) * (nx0 + 4) + (i + 0)] =
              rho[(j + 2) * (nx0 + 4) + (i + 0)];
          rhou0[(j - 1) * (nx0 + 4) + (i + 0)] =
              rhou0[(j + 1) * (nx0 + 4) + (i + 0)];
          rhou0[(j - 2) * (nx0 + 4) + (i + 0)] =
              rhou0[(j + 2) * (nx0 + 4) + (i + 0)];
          rhou1[(j - 1) * (nx0 + 4) + (i + 0)] =
              rhou1[(j + 1) * (nx0 + 4) + (i + 0)];
          rhou1[(j - 2) * (nx0 + 4) + (i + 0)] =
              rhou1[(j + 2) * (nx0 + 4) + (i + 0)];
          rhoE[(j - 1) * (nx0 + 4) + (i + 0)] =
              rhoE[(j + 1) * (nx0 + 4) + (i + 0)];
          rhoE[(j - 2) * (nx0 + 4) + (i + 0)] =
              rhoE[(j + 2) * (nx0 + 4) + (i + 0)];
        }
      }
      loop8 += std::chrono::high_resolution_clock::now() - start;
      // Bottom
      //writing dataset rho with (i, j+1), (i, j+2) access
      //reading dataset rho with (i, j-1), (i, j-2) access
      //writing dataset rhou0 with (i, j+1), (i, j+2) access
      //reading dataset rhou0 with (i, j-1), (i, j-2) access
      //writing dataset rhou1 with (i, j+1), (i, j+2) access
      //reading dataset rhou1 with (i, j-1), (i, j-2) access
      //writing dataset rhoE with (i, j+1), (i, j+2) access
      //reading dataset rhoE with (i, j-1), (i, j-2) access
      start = std::chrono::high_resolution_clock::now();
      #pragma acc parallel loop collapse (2)
      for (int j = nx1 + 1; j < nx1 + 2; j++) {
        for (int i = 0; i < nx0 + 4; i++) {
          rho[(j + 1) * (nx0 + 4) + (i + 0)] =
              rho[(j - 1) * (nx0 + 4) + (i + 0)];
          rho[(j + 2) * (nx0 + 4) + (i + 0)] =
              rho[(j - 2) * (nx0 + 4) + (i + 0)];
          rhou0[(j + 1) * (nx0 + 4) + (i + 0)] =
              rhou0[(j - 1) * (nx0 + 4) + (i + 0)];
          rhou0[(j + 2) * (nx0 + 4) + (i + 0)] =
              rhou0[(j - 2) * (nx0 + 4) + (i + 0)];
          rhou1[(j + 1) * (nx0 + 4) + (i + 0)] =
              rhou1[(j - 1) * (nx0 + 4) + (i + 0)];
          rhou1[(j + 2) * (nx0 + 4) + (i + 0)] =
              rhou1[(j - 2) * (nx0 + 4) + (i + 0)];
          rhoE[(j + 1) * (nx0 + 4) + (i + 0)] =
              rhoE[(j - 1) * (nx0 + 4) + (i + 0)];
          rhoE[(j + 2) * (nx0 + 4) + (i + 0)] =
              rhoE[(j - 2) * (nx0 + 4) + (i + 0)];
        }
      }
    }
    loop9 += std::chrono::high_resolution_clock::now() - start;
    // End of stage loop

    double sum = 0.0;
    double sum2 = 0.0;
    //reading dataset rho with (i, j) access
    //reading dataset p with (i, j) access
    start = std::chrono::high_resolution_clock::now();
    #pragma acc parallel loop collapse (2)
    for (int j = 0; j < nx1 + 4; j++) {
      for (int i = 0; i < nx0 + 4; i++) {
        sum += rho[j * (nx0 + 4) + i] * rho[j * (nx0 + 4) + i];
        sum2 += p[j * (nx0 + 4) + i] * p[j * (nx0 + 4) + i];
      }
    }
    loop10 += std::chrono::high_resolution_clock::now() - start;
    std::cout << "Checksums: " << sqrt(sum) << " " << sqrt(sum2) << "\n";

  } // End of time loop
} // pragma acc data close bracket

  // Record end time
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> diff = end - start;

  std::cout << "\nTimings are:\n";
  std::cout << "-----------------------------------------\n";
  // TODO: per-loop statistics come here
  std::cout << "Loop 1 time     " << loop1.count() / itercount << " seconds\n";
  std::cout << "Loop 2 time     " << loop2.count() / itercount / 3 << " seconds\n";
  std::cout << "Loop 3 time     " << loop3.count() / itercount / 3 << " seconds\n";
  std::cout << "Loop 4 time     " << loop4.count() / itercount / 3 << " seconds\n";
  std::cout << "Loop 5 time     " << loop5.count() / itercount / 3 << " seconds\n";
  std::cout << "Loop 6 time     " << loop6.count() / itercount / 3 << " seconds\n";
  std::cout << "Loop 7 time     " << loop7.count() / itercount / 3 << " seconds\n";
  std::cout << "Loop 8 time     " << loop8.count() / itercount / 3 << " seconds\n";
  std::cout << "Loop 9 time     " << loop9.count() / itercount / 3 << " seconds\n";
  std::cout << "Loop 10 time     " << loop10.count() / itercount / 3 << " seconds\n";
  std::cout << "Total Wall time " << diff.count() << " seconds\n";

  delete[] rho;
  delete[] rhou0;
  delete[] rhou1;
  delete[] rhoE;
  delete[] rho_old;
  delete[] rhou0_old;
  delete[] rhou1_old;
  delete[] rhoE_old;
  delete[] T;
  delete[] u0;
  delete[] u1;
  delete[] p;
  delete[] wk0;
  delete[] wk1;
  delete[] wk2;
  delete[] wk3;
}
