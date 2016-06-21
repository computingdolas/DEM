#include <cuda_runtime.h>
#include <vector>
#include "Type.h"

//cross product of 2 vectors of dimension 3 and store it in the second vector
__device__ void crossProd(const real_d *vec1, real_d *vec2){
    real_d temp[3];

    temp[0] = vec1[1]*vec2[2]-vec1[2]*vec2[1];
    temp[1] = vec2[0]*vec1[2]-vec1[0]*vec2[2];
    temp[2] = vec1[0]*vec2[1]-vec2[0]*vec1[1];

    for(int i=0;i<3;i++){
        vec2[i] = temp[i];
    }
}

//quat product of 2 quaternions and store them in the first vector
__device__ void quatProd(real_d *q1, const real_d* q2){
   real_d tq[4] = {q1[0],q1[1],q1[2],q1[3]};

   q1[0] = tq[0]*q2[0]-tq[1]*q2[1]-tq[2]*q2[2]-tq[3]*q2[3];
   q1[1] = tq[0]*q2[1]+tq[1]*q2[0]+tq[2]*q2[3]-tq[3]*q2[2];
   q1[2] = tq[0]*q2[2]-tq[1]*q2[3]+tq[2]*q2[0]+tq[3]*q2[1];
   q1[3] = tq[0]*q2[3]+tq[1]*q2[2]-tq[2]*q2[1]-tq[3]*q2[0];
}

//add 2 vectors of dimension 3 and store in the first
__device__ void add(real_d *vec1, const real_d *vec2){
    for(int i=0;i<3;i++){
        vec1[i] += vec2[i];
    }
}

//subtract 2 vectors of dimension 3 and store it in the first vector
__device__ void subtract(real_d *vec1, const real_d *vec2){
    for(int i=0;i<3;i++){
        vec1[i] -= vec2[i];
    }
}

//scalar multiply with a vector of dimension 3
__device__ void scalMult(real_d *vec, real_d scal){
    for(int i=0;i<3;i++){
        vec[i] = scal*vec[i];
    }
}

//compte dot product of 2 vectors
__device__ real_d dotProd(const real_d *vec1, const real_d *vec2){
    real_d sum = 0;
    for(int i=0;i<3;i++){
        sum += vec1[i]*vec2[i];
    }
    return sum;
}

//equalize left vector to right vector
__device__ void equalize(real_d *vec1, const real_d *vec2){
    for(int i=0;i<3;i++){
        vec1[i]= vec2[i];
    }
}

//Return globalID given the thread indices
__device__ u_int globalID(const u_int x, const u_int y, const u_int z, const int *numcells){
    return (z*(numcells[0]*numcells[1]) + y*numcells[0] + x);
}

__device__ bool isBoundaryCell(const u_int cell_id, const int *num_cells, bool *which_boundary){
    u_int ids[3];
    ids[0] =  (cell_id%(num_cells[0]*num_cells[1]))%num_cells[0];
    ids[1] = (cell_id%(num_cells[0]*num_cells[1]))/num_cells[0];
    ids[2] = cell_id/(num_cells[0]*num_cells[1]);

    bool ret_value = false;
    for(int i=0;i<3;i++){
        if(ids[i] == 0){
            which_boundary[2*i] = true;
            which_boundary[2*i+1] = false;
        }
        else if(ids[i] == num_cells[i]-1){
            which_boundary[2*i] = false;
            which_boundary[2*i+1] = true;
        }
        else{
            which_boundary[2*i] = false;
            which_boundary[2*i+1] = false;
        }

        ret_value = ret_value || which_boundary[2*i];
    }

    return ret_value;
}
//Give global cell ID of the cell in which the particle is located
__device__ u_int giveCellID(const real_d *position, const real_d *const_args, \
                            const int *num_cells, u_int *id, const  u_int idx){
    u_int vidx = idx*3;

    id[0] = (position[vidx]-const_args[0])/const_args[6];

    id[1] = (position[vidx+1]-const_args[2])/const_args[7];

    id[2] = (position[vidx+2]-const_args[4])/const_args[8];

    u_int cell_id = globalID(id[0],id[1],id[2],num_cells);

    return cell_id;
}

// Calculate the magnitude of the relative vector
__device__ real_d norm(const real_d *vector) {

    real_d sum = 0.0 ;
    for (int i =0 ; i < 3 ; ++i){
        sum  += vector[i] * vector[i] ;
    }
    return sqrt(sum) ;
}

//Contact detection
__device__ bool contactDetect(const u_int id_a, const u_int id_b, const real_d* position,\
                         const real_d *radius, real_d *pen_depth){

    u_int vid_a = id_a*3;
    u_int vid_b = id_b*3;

    real_d  rel_vec[3];
    for(int i=0;i<3;i++){
        rel_vec[i] = position[vid_a+i]-position[vid_b+i];
    }

    real_d vec_norm = norm(rel_vec);

    *pen_depth = radius[id_a]+radius[id_b]-vec_norm;
    if(*pen_depth < 0 ){
        return false;
    }
    return true;

}

__device__ void fillIterators(const u_int idx, const u_int idy,  const u_int idz, \
                               const int *num_cells, const u_int *reflect, \
                               int *I, int *J, int *K){
    if((idx == 0) && reflect[0]){
        I[0] = 0;
        I[1] = 1;
        I[2] = 10;
    }
    if((idx == num_cells[0]) && reflect[0]){
        I[2] = 10;
    }
    if((idy == 0) && reflect[1]){
        J[0] = 0;
        J[1] = 1;
        J[2] = 10;
    }
    if((idy == num_cells[1]) && reflect[1]){
        J[2] = 10;
    }
    if((idz == 0) && reflect[2]){
        K[0] = 0;
        K[1] = 1;
        K[2] = 10;
    }
    if((idz == num_cells[2]) && reflect[2]){
        K[2] = 10;
    }
}



__device__ void addForces(const u_int id_a, const u_int id_b, const real_d *position, real_d *force,\
                     real_d *temp_vel, real_d pen_depth, const real_d *const_args){
    const real_d kf = const_args[14];
    const real_d kdt = const_args[15];

    real_d normal[3];

    real_d force_n[3],vel_n[3];
    real_d force_t[3],vel_t[3];

    //printf("My position is %f %f %f\n",position[id_a*3],position[id_a*3+1],position[id_a*3+2]);
    equalize(normal,&position[id_b*3]);
    subtract(normal,&position[id_a*3]);

    scalMult(normal,-1.0);

    real_d norm_normal = norm(normal);

    //normal[] contains the unit vector along the normal
    if(norm_normal != 0){
        //printf("norm_normal not 0\n");
        scalMult(normal,(1.0/norm_normal));
    }
    equalize(vel_n,normal);
    real_d v_n = dotProd(normal,temp_vel);
    scalMult(vel_n,v_n);//vel_n is the normal component of velocity

    equalize(vel_t,vel_n);
    subtract(vel_t,temp_vel);
    scalMult(vel_t,-1.0);//vel_t contains the tangential velocity

    equalize(force_n,normal);
    scalMult(force_n,const_args[9]*pen_depth);// force_n += k_s*p (n^)
    scalMult(vel_n,-1.0*const_args[10]);//force_n  += kdn*vel_n
    add(force_n,vel_n);//force_n contains the normal component of force
    //printf("My normal is %f %f %f\n",force_n[0],force_n[1],force_n[2]);
    //printf("ks, kdn, p: %f %f %f",const_args[9],const_args[10],pen_depth);

    real_d ft = fmin(norm(force_n)*kf,kdt*norm(vel_t));

    if(norm(vel_t)  != 0){
       //printf("norm vel_t %f\n",norm(vel_t));
        scalMult(vel_t,(ft/norm(vel_t)));//vel_t now contains the tangential component of force
    }

    equalize(force_t,vel_t);//set force_t = vel_t

    //Finally add the computed forces
    add(&force[id_a*3],force_t);
    add(&force[id_a*3],force_n);

    //Return tangential force to temp_vel for torque calculation
    equalize(temp_vel,force_t);
    //printf("%f %f %f\n",force[id_a*3],force[id_a*3+1],force[id_a*3+2]);
}

__device__ void findContactVelocity(u_int idx, u_int curr_id, real_d *temp_pos1, \
                                    real_d *temp_pos2, const real_d *position,const \
                                    real_d *velocity, const real_d *radius, const real_d *a_velocity){
    //find contact position
     equalize(temp_pos1,&position[3*idx]);//x_a
     equalize(temp_pos2,&position[3*curr_id]);//x_b
     subtract(temp_pos1,temp_pos2);//x_a - x_b
     scalMult(temp_pos1,(radius[curr_id]/(radius[idx]+radius[curr_id])));//
     add(temp_pos1,temp_pos2);//temp_pos1 contains x

     //find contact velocity
     subtract(temp_pos2,temp_pos1); // x_b-x
     scalMult(temp_pos1,-1.0);//x-x_b
     subtract(temp_pos1,&position[3*idx]);//x - x_a

     crossProd(&a_velocity[idx*3],temp_pos1);//w_a*(x-x_a)
     crossProd(&a_velocity[curr_id*3],temp_pos2);//w_b*(x-x_b)

     add(temp_pos1,&velocity[3*idx]);//v_a + ....
     add(temp_pos2,&velocity[3*curr_id]);//v_b+.....

     subtract(temp_pos1,temp_pos2);//contact velocity is stored in temp_pos1
}

__device__ void positionCorrect(real_d *myposition, const real_d *const_args, const u_int *reflect){

    bool left,right;
    for(int i=0;i<3;i++){
        //check if I am out of domain bounds for each dimension
        right = myposition[i] > const_args[i*2+1];
        left = myposition[i] < const_args[i*2];
        if((right || left) && !reflect[i]){
                myposition[i] += left*(const_args[2*i+1]-const_args[2*i])-right*(const_args[2*i+1]-const_args[2*i]);
        }
    }
}

__global__ void initializeParticleList(u_int *particle_list, const u_int numparticles){
    u_int idx = blockDim.x*blockIdx.x+threadIdx.x;
    if(idx < numparticles){
        particle_list[idx] = idx+1;
    }
}

//Create the neighbour list of each cell
__global__ void createNeighbourList(int *neighbour_list, const int *num_cells, const u_int *reflect){
    u_int id_x = threadIdx.x+blockDim.x*blockIdx.x;
    u_int id_y = threadIdx.y+blockDim.y*blockIdx.y;
    u_int id_z = threadIdx.z+blockDim.z*blockIdx.z;

    u_int id_g = globalID(id_x,id_y,id_z,num_cells);
    u_int n_id = 26*id_g;

    //Iterators I,J and K....Will contain information related to neighbours in X,Y and Z directions
    int I[3] = {-1,0,1};
    int J[3] = {-1,0,1};
    int K[3] = {-1,0,1};
    if(id_x < num_cells[0] && id_y < num_cells[1] && id_z < num_cells[2]){
        fillIterators(id_x,id_y,id_z,num_cells,reflect,I,J,K);

        u_int temp_x,temp_y,temp_z;

        u_int count = 0;
        for(int i=0;i <3 && I[i] != 10;i++ ){
            for(int j=0;j <3 && J[j] != 10;j++ ){
                for(int k=0;k <3 && K[k] != 10;k++){
                    if(I[i] != 0 || J[j]!= 0  || K[k] != 0){
                        temp_x = (id_x+I[i]+num_cells[0])%num_cells[0];
                        temp_y = (id_y+J[j]+num_cells[1])%num_cells[1];
                        temp_z = (id_z+K[k]+num_cells[2])%num_cells[2];

                        neighbour_list[n_id+count] = globalID(temp_x,temp_y,temp_z,num_cells);
                        count++;
                    }
                }
            }
        }

        if(count < 26){
            neighbour_list[count] = id_g;
        }
    }

}

__global__ void initialiseRotation(real_d  * rotationVector,const u_int numparticles) {

    u_int idx  =   threadIdx.x+blockIdx.x*blockDim.x ;

    if (idx < numparticles){

        u_int vidx = idx * 3 ;
        rotationVector[vidx] = 1.0;
        rotationVector[vidx +1] = 0.0 ;
        rotationVector[vidx +2] = 0.0 ;


    }

}

__global__ void initialiseQuats(real_d * rotation,const u_int numparticles  ){

    u_int idx  =   threadIdx.x+blockIdx.x*blockDim.x ;

    if (idx < numparticles){

        u_int vidx = idx * 4 ;
        rotation[vidx] = 1.0 ;
        rotation[vidx+1] = 0.0 ;
        rotation[vidx +2] = 0.0 ;
        rotation[vidx +3] = 0.0 ;


    }
}

//Initialize the moment of inertia buffer
__global__ void initializeMoi(real_d *moi, const real_d *radius, const real_d *mass, const u_int numparticles){
    u_int idx = threadIdx.x+ blockIdx.x*blockDim.x;

    if(idx < numparticles && !isinf(radius[idx])){
        moi[idx] = 0.4*mass[idx]*radius[idx]*radius[idx];

    }
}

//Update the angular velocity
__global__ void updateAngVelocity(real_d *a_velocity,const real_d *moi,const real_d *torque,\
                                  const real_d time_step, const u_int numparticles){
    u_int idx = threadIdx.x+ blockIdx.x*blockDim.x;

    if(idx < numparticles && moi[idx] != 0){
        for(int i=0;i<3;i++){
            a_velocity[idx*3+i] += time_step*(torque[idx*3+i])/moi[idx];
        }
    }

}

//Update the quaternion
__global__ void updateQuat(real_d *rotation,real_d *a_velocity,const real_d time_step, const real_d numparticles){
    u_int idx = threadIdx.x+blockIdx.x*blockDim.x;


    if(idx < numparticles){
        real_d omg_q[4] = {0.0,a_velocity[idx*3],a_velocity[idx*3+1],a_velocity[idx*3+2]};
        quatProd(omg_q,&rotation[idx*4]);//quaternion: (0,w)*q stored in omg_q
        for(int i=0;i<4;i++){
            rotation[idx*4+i] += 0.5*time_step*omg_q[i];
        }
    }
}

//Apply quaternion to find the rotation
__global__ void applyQuats(real_d *rotvec, const real_d *rot, const u_int numparticles){
   u_int idx = threadIdx.x+blockIdx.x*blockDim.x;


   if(idx < numparticles){
       real_d qt[4] = {rot[4*idx],-1.0*rot[4*idx+1],-1.0*rot[4*idx+2],-1.0*rot[4*idx+3]};
       real_d q[4] = {rot[4*idx],rot[4*idx+1],rot[4*idx+2],rot[4*idx+3]};
       real_d r[4] = {0.0,rotvec[3*idx],rotvec[3*idx+1],rotvec[3*idx+2]};

       quatProd(r,qt);
       quatProd(q,r);

       for(int i=0;i<3;i++){
           rotvec[3*idx+i] = q[i+1];
       }

   }
}

//Update quaternions
// Update the list particle parallely
__global__ void updateListsParPar(u_int * cell_list, u_int * particle_list, const real_d  * const_args, const u_int num_particles ,const  real_d * position, const int * num_cells  ) {

    u_int idx = threadIdx.x+blockIdx.x*blockDim.x;

    if (idx < num_particles) {

        // Finding the index of the particles
        u_int  vidxp = idx * 3 ;

        // Finding the cordinates of the particle
        real_d pos[3] = {position[vidxp], position[vidxp+1], position[vidxp+2] } ;

        // Find the ... cordinate of the cell it lies and register it their using atomic operations
        u_int i = (pos[0]-const_args[0]) / const_args[6] ;
        u_int j = (pos[1]-const_args[2]) / const_args[7] ;
        u_int k = (pos[2]-const_args[4])/ const_args[8];

        // Find the global id of the cell
        u_int cellindex = globalID(i,j,k,num_cells) ;
      //printf("%f %f %f %u\n",pos[0],pos[1],pos[2],cellindex);
        // See whether that cell has already has some master particle , and if not assign itself to it and
        u_int old = atomicExch(&cell_list[cellindex] ,idx+1);

        particle_list[idx] = old ;
    }
}

//Wall contact forces
//Contact detection and force calculation
__global__ void calcForces(real_d *force, real_d *torque, const real_d *position,const real_d *mass, \
                           const real_d *radius, const real_d *const_args, const int* num_cells, const u_int *reflect,\
                           const u_int *cell_list, const u_int *particle_list, const int* neighbour_list, \
                           const real_d *velocity, const real_d *a_velocity, const u_int numparticles){

    u_int idx = threadIdx.x+blockDim.x*blockIdx.x;

    //global cell_id
    u_int cell_id;

    //where the neighbour list starts
    u_int n_id,count;

    //cell indices in 3d
    u_int id[3];

    int head_id;

    real_d gravity[3] = {const_args[11],const_args[12],const_args[13]};
    real_d temp_vel[3],temp_pos1[3],temp_pos2[3],temp_force[3],pen_depth;
    real_d cont_pos[3],tang_force[3];

    bool in_contact;
    bool which_boundary[6];

    u_int bound_id = 0;
    if(idx < numparticles && !isinf(mass[idx])){
        force[idx*3] = gravity[0];
        force[idx*3+1] = gravity[1];
        force[idx*3+2] = gravity[2];

        torque[idx*3] = 0.0;
        torque[idx*3+1] = 0.0;
        torque[idx*3+2] = 0.0;
        cell_id = giveCellID(position,const_args,num_cells,id,idx);

        n_id = 26*cell_id;//for indexing the neighbour list

        //Check contact with walls and add forces accordingly
        if(isBoundaryCell(cell_id,num_cells,which_boundary)){
            real_d wall_pos,vn,mag;
            for(int i=0;i<3;i++){
                if((which_boundary[2*i] || which_boundary[2*i+1]) && reflect[i]){
                   bound_id = which_boundary[2*i+1];
                   wall_pos = const_args[2*i+bound_id];//position of the wall

                   //Detect contact with wall
                   pen_depth = radius[idx]-abs(wall_pos-position[idx*3+i]);
                   if(pen_depth < 0){
                       //(x - x_a)
                       temp_pos1[i] = (wall_pos-position[idx*3+i]);temp_pos1[(i+1)%3]=0;temp_pos1[(i+2)%3]=0;
                       equalize(temp_pos2,temp_pos1);

                       if(norm(temp_pos2) != 0){
                         scalMult(temp_pos2,(1.0/norm(temp_pos2)));//unit normal vector
                       }
                       equalize(temp_force,temp_pos2);

                       //w*(x - x_a)
                       crossProd(&a_velocity[idx*3],temp_pos1);

                       // v += ..... Contact velocity
                       add(temp_pos1,&velocity[idx*3]);

                       //storing v_n in temp_pos2
                       vn = dotProd(temp_pos1,temp_pos2);
                       scalMult(temp_pos2,vn);

                       //storing v_t in temp_pos1
                       subtract(temp_pos1,temp_pos2);

                       //kdn*v_n
                       scalMult(temp_pos2,-1.0*const_args[10]);

                       //k_s*pn
                       scalMult(temp_force,const_args[9]*pen_depth);

                       //compile the normal force
                       add(temp_force,temp_pos2);//temp_force = F_n


                       //Find the magnitude of tangential force
                       mag =  fmin(const_args[14]*norm(temp_pos1),const_args[15]*norm(temp_force));

                       //store F_t in temp_pos1
                       if(norm(temp_pos1) != 0.0){
                        scalMult(temp_pos1,(mag/norm(temp_pos1)));
                       }


                       //add tangential forces
                       add(&force[idx*3],temp_pos1);
                       //printf("%f %f %f\n",temp_pos1);
                       //add normal forces
                       add(&force[idx*3],temp_force);

                       //Torque Calculation
                       for(int j=0;j<3;j++){
                           cont_pos[j] = (j == i)?(wall_pos-position[idx*3+j]):(-1.0*position[idx*3+j]);
                       }
                       equalize(tang_force,temp_pos1);
                       crossProd(cont_pos,tang_force);
                       add(torque,tang_force);
                   }

                }
            }
         }


        //Calculate the forces...First iterate through cells which are neighbours
        count=0;
        while((count < 26) && (neighbour_list[n_id+count] != cell_id)){
            //iterate through the particle list of this cell
            head_id = cell_list[neighbour_list[n_id+count]]-1;
            for(int curr_id = head_id;curr_id != -1;curr_id = particle_list[curr_id]-1){
               if(curr_id != idx){
                   //Contact detection
                   in_contact =  contactDetect(idx,curr_id,position,radius,&pen_depth);
                   if(in_contact){
                       //contact velocity is computed and stored in temp_pos1
                       findContactVelocity(idx,curr_id,temp_pos1,temp_pos2,position,\
                                           velocity,radius,a_velocity);
                       equalize(temp_vel,temp_pos1);
                       addForces(idx,curr_id,position,force,temp_vel,pen_depth,const_args);

                       //torque calculation
                       for(int j=0;j<3;j++){
                           cont_pos[j] = position[curr_id*3+j]\
                                         +(radius[curr_id]/\
                                         (radius[curr_id]+radius[idx]))*(position[curr_id*3+j]-\
                                                                         position[idx*3+j]);
                       }

                       equalize(tang_force,temp_vel);
                       crossProd(cont_pos,tang_force);
                       add(torque,tang_force);
                   }
               }

            }
            count++;
        }


        //Now iterate through own list
        head_id = cell_list[cell_id]-1;
        for(int curr_id = head_id;curr_id != -1;curr_id  = particle_list[curr_id]-1){
            if(curr_id != idx){
                //Contact detection
                in_contact =  contactDetect(idx,curr_id,position,radius,&pen_depth);
                if(in_contact){
                    //contact velocity is computed and stored in temp_pos1
                    findContactVelocity(idx,curr_id,temp_pos1,temp_pos2,position,\
                                        velocity,radius,a_velocity);
                    equalize(temp_vel,temp_pos1);
                    addForces(idx,curr_id,position,force,temp_vel,pen_depth,const_args);

                    //torque calculation
                    for(int j=0;j<3;j++){
                        cont_pos[j] = position[curr_id*3+j]\
                                      +(radius[curr_id]/\
                                      (radius[curr_id]+radius[idx]))*(position[curr_id*3+j]-\
                                                                      position[idx*3+j]);
                    }

                    equalize(tang_force,temp_vel);
                    crossProd(cont_pos,tang_force);
                    add(torque,tang_force);
                }
            }
        }

    }
}



//Position update
__global__ void updatePosition(const real_d *force,real_d *position,const real_d* velocity, \
                               const real_d * mass,const u_int numparticles,const real_d timestep,\
                               const real_d* const_args, const u_int *reflect) {

    u_int idx = threadIdx.x + blockIdx.x * blockDim.x ;

    if(idx < numparticles && !isinf(mass[idx])){
        u_int vidx = idx * 3 ;

	//printf("%f %f %f %f\n",position[vidx],position[vidx+1],position[vidx+2],mass[idx]);

        position[vidx]   += (timestep * velocity[vidx] ) + ( (force[vidx] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;
        position[vidx+1] += (timestep * velocity[vidx+1] ) + ( (force[vidx+1] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;
        position[vidx+2] += (timestep * velocity[vidx+2] ) + ( (force[vidx+2] * timestep * timestep) / ( 2.0 * mass[idx]) ) ;

    //printf("%f %f %f %f\n",force[vidx],force[vidx+1],force[vidx+2],mass[idx]);
        // printf("%f %f %f\n",force[idx*3],force[idx*3+1],force[idx*3+2]);
        //Check for boundary conditions and correct the positions accordingly
        positionCorrect(&position[vidx],const_args,reflect);
	//printf("%f %f %f %f\n",position[vidx],position[vidx+1],position[vidx+2],mass[idx]);
    }
}

// Copying the forces
__global__ void copyForces(real_d * fold,real_d * fnew, real_d *told, real_d *tnew,\
                           const u_int numparticles) {

    u_int idx = threadIdx.x + blockIdx.x * blockDim.x ;

    if(idx < numparticles){
        u_int vidxp = idx * 3 ;

        for(u_int i =vidxp ; i < vidxp+3; ++i ){
            fold[i] = fnew[i] ;
            told[i] = told[i];
        }
    }
}

//Reset the buffer with 0s if the thread grid is 3 dimensional
template <typename T>__global__ void setToZero3D(T *buffer, const int* num_cells){
    u_int idx = threadIdx.x+blockIdx.x*blockDim.x;
    u_int idy = threadIdx.y+blockIdx.y*blockDim.y;
    u_int idz = threadIdx.z+blockIdx.z*blockDim.z;

    u_int id_g = globalID(idx,idy,idz,num_cells);
    if(idx < num_cells[0] && idy < num_cells[1] && idz < num_cells[2]){
        buffer[id_g] = 0;
    }
}

// Velocity Update
__global__ void updateVelocity(const real_d*forceNew,const real_d*forceOld,real_d * velocity, \
                               const real_d* mass,const u_int numparticles ,const real_d timestep ){

    u_int idx = threadIdx.x + blockIdx.x * blockDim.x ;

    if(idx < numparticles){
        u_int vidx = idx * 3 ;

        velocity[vidx] += ( (forceNew[vidx] + forceOld[vidx]) * timestep ) / (2.0 * mass[idx] ) ;
        velocity[vidx+1] += ( (forceNew[vidx+1] + forceOld[vidx+1]) * timestep ) / (2.0 * mass[idx] ) ;
        velocity[vidx+2] += ( (forceNew[vidx+2] + forceOld[vidx+2]) * timestep ) / (2.0 * mass[idx] ) ;
    }

}



