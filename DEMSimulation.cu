#include <iostream>
#include <list>
#include "cudaDeviceBuffer.h"
#include <cuda_runtime.h>
#include "Parser.h"
#include "PhysicalVariable.h"
#include "Type.h"
#include "kernels.cuh"
#include <string>
#include "VTKWriter.h"
#include <iomanip>
#include "Time.hpp"


int main(int argc, char *argv[]){
    //Read parameters from file and store it in the parser object. Also read in the initial configuration
    Parser p(argv[1]);
    p.readParameters();
    p.readInputConfiguration();

    //time stepping parameters
    real_d time_end = std::stod(p.params["time_end"]) ;
    real_d timestep_length = std::stod(p.params["timestep_length"]) ;

    //vtk file output parameters
    u_int vtk_out_freq = std::stol(p.params["vtk_out_freq"]) ;
    std::string vtk_name = p.params["vtk_out_name_base"] ;

    //Kernel launch parameters
    u_int threads_per_blocks = std::stol(p.params["cl_workgroup_1dsize"]) ;
    u_int threads_per_blocks_x = std::stol(p.params["cl_workgroup_3dsize_x"]);
    u_int threads_per_blocks_y = std::stol(p.params["cl_workgroup_3dsize_y"]);
    u_int threads_per_blocks_z = std::stol(p.params["cl_workgroup_3dsize_z"]);

    //Domain parameters
    real_d xmin = std::stod(p.params["x_min"]);
    real_d xmax = std::stod(p.params["x_max"]);
    real_d ymin = std::stod(p.params["y_min"]);
    real_d ymax = std::stod(p.params["y_max"]);
    real_d zmin = std::stod(p.params["z_min"]);
    real_d zmax = std::stod(p.params["z_max"]);
    u_int xn = std::stol(p.params["x_n"]);
    u_int yn = std::stol(p.params["y_n"]);
    u_int zn = std::stol(p.params["z_n"]);

    //gravity, spring and damping parameters
    real_d gx = std::stod(p.params["g_x"]);
    real_d gy = std::stod(p.params["g_y"]);
    real_d gz = std::stod(p.params["g_z"]);
    real_d ks = std::stod(p.params["k_s"]);
    real_d kdn = std::stod(p.params["k_dn"]);

    //number of particles and cells
    const u_int numparticles = p.num_particles ;
    const u_int numcells = xn*yn*zn;

    //Creating necessary buffers
    cudaDeviceBuffer<real_d> mass(numparticles,PhysicalQuantity::Scalar) ;
    cudaDeviceBuffer<real_d> radius(numparticles,PhysicalQuantity::Scalar);
    cudaDeviceBuffer<real_d> position(numparticles,PhysicalQuantity::Vector);
    cudaDeviceBuffer<real_d> a_velocity(numparticles,PhysicalQuantity::Vector);
    cudaDeviceBuffer<real_d> velocity(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> forceold(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> forcenew(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> torqueold(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> torquenew(numparticles,PhysicalQuantity::Vector) ;
    cudaDeviceBuffer<real_d> moi(numparticles,PhysicalQuantity::Scalar);

    // Domain Buffers
    cudaDeviceBuffer<u_int> cell_list(numcells,PhysicalQuantity::Scalar);
    cudaDeviceBuffer<u_int> particle_list(numparticles,PhysicalQuantity::Scalar);
    cudaDeviceBuffer<real_d> const_args(16,PhysicalQuantity::Scalar);
    cudaDeviceBuffer<int> num_cells(3,PhysicalQuantity::Scalar);
    cudaDeviceBuffer<int> neighbour_list(26*numcells,PhysicalQuantity::Scalar);
    cudaDeviceBuffer<u_int> reflect(3,PhysicalQuantity::Scalar);

    // for rotation
    cudaDeviceBuffer<real_d> rotation(numparticles,PhysicalQuantity::Quat);
    cudaDeviceBuffer<real_d> rotationvector(numparticles, PhysicalQuantity::Vector) ;

    //reflecting or periodic boundaries
    reflect[0] = std::stol(p.params["reflect_x"]);
    reflect[1] = std::stol(p.params["reflect_y"]);
    reflect[2] = std::stol(p.params["reflect_z"]);

    real_d len_x = (xmax-xmin)/xn;
    real_d len_y = (ymax-ymin)/yn;
    real_d len_z = (zmax-zmin)/zn;

    //Initiliazing the buffers for mass,velocity and position
    p.fillBuffers(mass,radius,velocity,position);

    //Filling in the host data for the constant arguements
    const_args[0] = xmin;
    const_args[1] = xmax;
    const_args[2] = ymin;
    const_args[3] = ymax;
    const_args[4] = zmin;
    const_args[5] = zmax;
    const_args[6] = len_x;
    const_args[7] = len_y;
    const_args[8] = len_z;
    const_args[9] = ks;
    const_args[10] = kdn;
    const_args[11] = gx;
    const_args[12] = gy;
    const_args[13] = gz;
    const_args[14] = 0.1;//kf
    const_args[15] = 0.1;//kdt

    //Number of cells per dimension
    num_cells[0] = xn;
    num_cells[1] = yn;
    num_cells[2] = zn;

    // Allocating memory on Device
    mass.allocateOnDevice();
    radius.allocateOnDevice();
    position.allocateOnDevice();

    a_velocity.allocateOnDevice();
    velocity.allocateOnDevice();
    forceold.allocateOnDevice();
    forcenew.allocateOnDevice();
    torqueold.allocateOnDevice();
    torquenew.allocateOnDevice();
    moi.allocateOnDevice();
    cell_list.allocateOnDevice();
    particle_list.allocateOnDevice();
    const_args.allocateOnDevice();
    num_cells.allocateOnDevice();
    reflect.allocateOnDevice();
    neighbour_list.allocateOnDevice();

    // for rotation
    rotation.allocateOnDevice();
    rotationvector.allocateOnDevice();

    //Copy to Device
    mass.copyToDevice();
    radius.copyToDevice();
    position.copyToDevice();
    velocity.copyToDevice();
    a_velocity.copyToDevice();
    forceold.copyToDevice();
    forcenew.copyToDevice();
    torquenew.copyToDevice();
    torqueold.copyToDevice();
    moi.copyToDevice();

    // domain buffer copied to device
    cell_list.copyToDevice();
    particle_list.copyToDevice();
    const_args.copyToDevice();
    num_cells.copyToDevice();
    reflect.copyToDevice();
    neighbour_list.copyToDevice();

    // rotation buffer copied to devie
    rotationvector.copyToDevice();
    rotation.copyToDevice();

    VTKWriter writer(vtk_name) ;

    //Calculate the number of blocks
    //Calculate the number of blocks for both types of launches(1D and 3D)
    u_int num_blocks,num_blocks_x,num_blocks_y,num_blocks_z ;

    if(numparticles % threads_per_blocks ==0)
        num_blocks = numparticles / threads_per_blocks ;
    else
        num_blocks = (numparticles / threads_per_blocks) + 1;

    if(num_cells[0] % threads_per_blocks_x ==0)
        num_blocks_x = num_cells[0] / threads_per_blocks_x ;
    else
        num_blocks_x = (num_cells[0] / threads_per_blocks_x) + 1;

    if(num_cells[1] % threads_per_blocks_y ==0)
        num_blocks_y = num_cells[1] / threads_per_blocks_y ;
    else
        num_blocks_y = (num_cells[1] / threads_per_blocks_y) + 1;

    if(num_cells[2] % threads_per_blocks_y ==0)
        num_blocks_z = num_cells[2] / threads_per_blocks_z ;
    else
        num_blocks_z = (num_cells[2] / threads_per_blocks_z) + 1;

    dim3 blockDim(threads_per_blocks_x,threads_per_blocks_y,threads_per_blocks_z);
    dim3 gridDim(num_blocks_x,num_blocks_y,num_blocks_z);

    real_d time_taken = 0.0 ;

    HESPA::Timer time ;
    // Algorithm to follow
    {

        u_int iter = 0 ;

        //Initialize moment of inertia for all particles
        initializeMoi<<<num_blocks,threads_per_blocks>>>(moi.devicePtr,radius.devicePtr,mass.devicePtr,numparticles);

        //Initialise quats for every particle
        initialiseQuats<<<num_blocks,threads_per_blocks >>>  (rotation.devicePtr,numparticles);

        // Initialising all the rotation for every particle
        initialiseRotation<<< num_blocks,threads_per_blocks >>>  (rotationvector.devicePtr,numparticles) ;

        //Create the list of neighbours for each cell depending upon the boundary conditions
        createNeighbourList<<<gridDim,blockDim>>>(neighbour_list.devicePtr,num_cells.devicePtr,reflect.devicePtr);

        //Ready the particle and cell list for updates
        initializeParticleList<<<num_blocks,threads_per_blocks>>>(particle_list.devicePtr,numparticles);

        //Update the linked list
        updateListsParPar<<<num_blocks,threads_per_blocks>>>(cell_list.devicePtr,particle_list.devicePtr,const_args.devicePtr,\
                                                             numparticles,position.devicePtr,num_cells.devicePtr);

        //Calculate initial forces
        calcForces<<<num_blocks,threads_per_blocks>>>(forcenew.devicePtr,torquenew.devicePtr,position.devicePtr,mass.devicePtr,radius.devicePtr, \
                                                     const_args.devicePtr,num_cells.devicePtr, reflect.devicePtr,\
                                                     cell_list.devicePtr,particle_list.devicePtr,neighbour_list.devicePtr, \
                                                     velocity.devicePtr,a_velocity.devicePtr, numparticles,iter);



        //Start time stepping now
        for(real_d t =0.0 ; t < time_end; t+= timestep_length ) {
            time.reset();

            if(iter % vtk_out_freq == 0){
                // copy to host back
                forcenew.copyToHost();
                forceold.copyToHost();
                position.copyToHost();
                velocity.copyToHost();
                writer.writeVTKOutput(mass,radius,position,velocity,rotationvector,numparticles);
            }

            // Update the Position
            updatePosition<<<num_blocks,threads_per_blocks>>>(forcenew.devicePtr,position.devicePtr,velocity.devicePtr,\
                                                              mass.devicePtr,numparticles,timestep_length,\
                                                              const_args.devicePtr,reflect.devicePtr);

            //update angular velocity
            updateAngVelocity<<<num_blocks,threads_per_blocks>>>(a_velocity.devicePtr,moi.devicePtr,torquenew.devicePtr,\
                                                                 timestep_length,numparticles);

            //update quaternion
            updateQuat<<<num_blocks,threads_per_blocks>>>(rotation.devicePtr,a_velocity.devicePtr,timestep_length,numparticles);

            //apply quaternion to the rotation
            applyQuats<<<num_blocks,threads_per_blocks>>>(rotationvector.devicePtr,rotation.devicePtr,numparticles);

            // Copy the forces
            copyForces<<<num_blocks,threads_per_blocks>>>(forceold.devicePtr,forcenew.devicePtr, \
                                                          torqueold.devicePtr,torquenew.devicePtr, numparticles);

            //Ready the particle and cell list for updates
            initializeParticleList<<<num_blocks,threads_per_blocks>>>(particle_list.devicePtr,numparticles);
            //Reset the linked lists to 0
            setToZero3D<<<gridDim,blockDim>>>(cell_list.devicePtr,num_cells.devicePtr);
            //Update the linked list
            updateListsParPar<<<num_blocks,threads_per_blocks>>>(cell_list.devicePtr,particle_list.devicePtr,const_args.devicePtr,\
                                                                 numparticles,position.devicePtr,num_cells.devicePtr);


            //Calculate forces
            calcForces<<<num_blocks,threads_per_blocks>>>(forcenew.devicePtr,torquenew.devicePtr, position.devicePtr,mass.devicePtr,radius.devicePtr, \
                                                         const_args.devicePtr, num_cells.devicePtr, reflect.devicePtr,\
                                                         cell_list.devicePtr, particle_list.devicePtr,neighbour_list.devicePtr,\
                                                         velocity.devicePtr, a_velocity.devicePtr, numparticles,iter);


            // Update the velocity
            updateVelocity<<<num_blocks,threads_per_blocks>>>(forcenew.devicePtr,forceold.devicePtr,velocity.devicePtr,\
                                                              mass.devicePtr,numparticles,timestep_length);
            time_taken += time.elapsed();

            // Iterator count
            ++iter ;

        }


    }

    std::cout<<"The time taken for "<<numparticles<<" is:= "<<time_taken<<std::endl ;

    return 0;


}
